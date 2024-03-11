import argparse
import os
import torch
from transformers import  AutoModelForCausalLM, AutoTokenizer
import onnx

def export(args):
    id = args.model_id
    out_path = args.out_dir +os.sep+ "model.onnx"
    logits_out_path = args.out_dir +os.sep+ "logits.onnx"
    model = AutoModelForCausalLM.from_pretrained(id,  trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(id, trust_remote_code=True)
    inputs = tokenizer("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat's the weather?<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt")

    all_ids = inputs["input_ids"]

    seq_len = len(inputs["input_ids"][0])

    model_e = model.eval()
    class Qwen2Model(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids,position_ids,past_key_values):
            return self.m(
                input_ids = input_ids,
                position_ids= position_ids,
                attention_mask = None,
                past_key_values = past_key_values,
                use_cache = True,
                )

    m = Qwen2Model(model_e)

    batch = 1
    N = 1

    input_ids = torch.ones([batch, N], dtype=torch.int64)
    position_ids = torch.arange(0,seq_len ).unsqueeze(0)

    dynamic_axes = {
        'input_ids': {1: 'N', },
        'position_ids': {1: 'N', },
    }

    past_key_values=[]

    kv_cache_dyn_axes = {2: "N"}
    in_names = ["input_ids","position_ids"]
    out_names = ["hidden_states"]
    past_key_values = []
    for i in range(24):
        past_key_in = torch.randn([1,16,0,64])
        past_value_in = torch.randn([1,16,0,64])
        in_names.extend([f"past_key_in{i}", f"past_value_in{i}"])
        out_names.extend([f"past_key_out{i}", f"past_value_out{i}"])
        dynamic_axes[f"past_key_in{i}"] = kv_cache_dyn_axes
        dynamic_axes[f"past_value_in{i}"] = kv_cache_dyn_axes
        past_key_values.append((past_key_in, past_value_in))


        
    torch.onnx.export(
        m,
        (all_ids,position_ids,past_key_values,),
        out_path,
        opset_version=17,
        do_constant_folding=False,
        input_names=in_names,
        output_names=out_names,
        dynamic_axes=dynamic_axes,
        verbose = False,
    )


    model = onnx.load(out_path)


    print("Model Inputs:")
    for input in model.graph.input:
        input_name = input.name
        input_shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
        print(f"Name: {input_name}, Shape: {input_shape}, Type: {onnx.TensorProto.DataType.Name(input.type.tensor_type.elem_type)}")

    print("\nModel Outputs:")
    for output in model.graph.output:
        output_name = output.name
        output_shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        print(f"Name: {output_name}, Shape: {output_shape}, Type: {onnx.TensorProto.DataType.Name(output.type.tensor_type.elem_type)}")
    

    def repetitionPenaltyLogitsProcess(input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
     penalty = 1.1
     score = torch.gather(scores, 1, input_ids)
     # if score < 0 then repetition penalty has to be multiplied to reduce the token probabilities
     score = torch.where(score < 0, score * penalty, score / penalty)
     scores.scatter_(1, input_ids, score)
     return scores


    def topK(scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = 50
        filter_value = -float("Inf")
        top_k = min(top_k, scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(indices_to_remove, filter_value)
        return scores

    def topP(scores: torch.FloatTensor) -> torch.FloatTensor:
        top_p = 0.8
        filter_value = -float("Inf")
        min_tokens_to_keep = 1
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -min_tokens_to_keep :] = 0
        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, filter_value)
        return scores

    def processLogits(input_ids, next_token_logits):
        next_token_scores = repetitionPenaltyLogitsProcess(input_ids, next_token_logits)
        next_token_scores = topK(next_token_scores)
        next_token_scores = topP(next_token_scores)
        probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        return next_tokens
    
    class LogitsModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self,all_input_ids, logits):
            return processLogits(all_input_ids,logits[:, -1, :])
        

    m = LogitsModel()
    dynamic_axes = {
        'all_input_ids': {1: 'X', },
    }
    logits = torch.randn([1,1,151936])


    torch.onnx.export(
        m,
        (all_ids,logits),
        logits_out_path,
        opset_version=17,
        do_constant_folding=False,
        input_names=["all_input_ids","logits"],
        output_names=["token_id"],
        dynamic_axes=dynamic_axes,
        verbose = False,
    )

    model = onnx.load(logits_out_path)

    print("Logits Model Inputs:")
    for input in model.graph.input:
        input_name = input.name
        input_shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
        print(f"Name: {input_name}, Shape: {input_shape}, Type: {onnx.TensorProto.DataType.Name(input.type.tensor_type.elem_type)}")

    print("\nLogits Model Outputs:")
    for output in model.graph.output:
        output_name = output.name
        output_shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        print(f"Name: {output_name}, Shape: {output_shape}, Type: {onnx.TensorProto.DataType.Name(output.type.tensor_type.elem_type)}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='export qwen1.5',
    )
    parser.add_argument('-m', '--model_id', required=True, type=str)
    parser.add_argument('-o', '--out_dir', required=True, type=str)
    args = parser.parse_args()
 
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
 
    export(args)