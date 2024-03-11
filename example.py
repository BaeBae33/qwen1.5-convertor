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
logis_session = ort.InferenceSession(sys.argv[2] )

def generate(input_ids, position_ids, past_key_values):
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
    # return (processLogits(all_ids,torch.tensor(outputs[0])[:, -1, :]),outputs[1:])
    return logis_session.run(None,{
        "all_input_ids":all_ids.numpy(),
        "logits":outputs[0],
    }), outputs[1:]


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
    next_token_id, past_key_values = generate( input_ids, position_ids, past_key_values)
    if next_token_id == 151643:
        break
    all_ids = torch.cat((all_ids, torch.tensor(next_token_id)), dim=-1)
    seq_len=seq_len+1
    input_ids =  next_token_id
    position_ids = [[seq_len-1]]

text = tokenizer.batch_decode(all_ids)[0]
print(text)

    

