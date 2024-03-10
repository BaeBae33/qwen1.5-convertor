import argparse
import os
import torch
from transformers import  AutoModelForCausalLM, AutoTokenizer
import onnx

def export(args):
    id = args.model_id
    out_path = args.out_dir +os.sep+ "model.onnx"
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