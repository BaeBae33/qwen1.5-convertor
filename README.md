# qwen1.5-convertor
convert qwen1.5 to onnx or tflite

## Convert-Or-Download
You can convert model or download from huggingface directly
### convert
```
pip install git+https://github.com/BaeBae33/transformers-for-qwen1.5-export.git
python convert.py -m Qwen/Qwen1.5-0.5B-Chat -o ~/Downloads/qwen-onnx
```
### download
```
https://huggingface.co/baebae/Qwen1.5-0.5B-Chat-ONNX
```

## Usage
```
python example.py ~/Downloads/qwen-onnx/model.onnx
```

