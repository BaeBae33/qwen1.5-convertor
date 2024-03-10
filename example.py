import sys
import torch
import numpy as np
import onnxruntime as ort
from transformers import  AutoModelForCausalLM, AutoTokenizer

id = "Qwen/Qwen1.5-0.5B-Chat"
model = AutoModelForCausalLM.from_pretrained(id,  trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(id, trust_remote_code=True)
inputs = tokenizer("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat's the weather?<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt")

all_ids = inputs["input_ids"]
session = ort.InferenceSession(sys.argv[1] )

def repetitionPenaltyLogitsProcess(input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
     penalty = 1.1
     score = torch.gather(scores, 1, input_ids)
     # if score < 0 then repetition penalty has to be multiplied to reduce the token probabilities
     score = torch.where(score < 0, score * penalty, score / penalty)
     scores.scatter_(1, input_ids, score)
     return scores


def topK(input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    top_k = 50
    filter_value = -float("Inf")
    top_k = min(top_k, scores.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
    scores = scores.masked_fill(indices_to_remove, filter_value)
    return scores

def topP(input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
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
    next_token_scores = topK(input_ids, next_token_scores)
    next_token_scores = topP(input_ids, next_token_scores)
    probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
    return next_tokens


def generate(seq_len, input_ids, position_ids, past_key_values):
    inputs = {
        "input_ids":input_ids,
        "position_ids":position_ids
    }
    for i in range(24):
        key_index = 2*i
        value_index = key_index+1
        inputs["past_key_in"+ str(i)] = past_key_values[key_index]
        inputs["past_value_in" + str(i)] =past_key_values[value_index]
    outputs = session.run(None,inputs)
    return (processLogits(all_ids,torch.tensor(outputs[0])[:, -1, :]),outputs[1:])


max_len = 50
seq_len = len(all_ids[0])

past_key_values=[]

for i in range(48):
    past_key_values.append(np.zeros((1,16,0,64), np.float32))
position_ids = torch.arange(0,seq_len ).unsqueeze(0).numpy()

input_ids = all_ids.numpy()

for i in range(max_len):
    if seq_len>=max_len:
        break
    next_token_id, past_key_values = generate(seq_len, input_ids, position_ids, past_key_values)
    if next_token_id == 151643:
        break
    all_ids = torch.cat((all_ids, torch.tensor([[next_token_id]])), dim=-1)
    seq_len=seq_len+1
    input_ids =  [[next_token_id]]
    position_ids = [[seq_len-1]]

text = tokenizer.batch_decode(all_ids)[0]
print(text)

    

