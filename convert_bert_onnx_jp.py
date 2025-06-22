import torch
from torch import nn

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnx
from onnxsim import simplify
from onnxconverter_common import float16 as float16_converter

class DisableCompileContextManager:
    def __init__(self):
        self._original_compile = torch.compile

    def __enter__(self):
        # Turn torch.compile into a no-op
        torch.compile = lambda *args, **kwargs: lambda x: x

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.compile = self._original_compile

def export():
    onnx_temp_model_path= "./bert/ruri-v3-30m-onnx/model_fp16.onnx"
    #'./model.onnx'
    onnx_fp16_model_path = "./bert/ruri-v3-30m-onnx/model_fp16.onnx"

    class ONNXBert(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "./bert/ruri-v3-30m",
                num_labels=3,
                # Flash Attention 2.0's recalculated memory access pattern and partitioning strategy causes
                # onnx to report an error by not being able to compute the export map
                attn_implementation="eager"  # Use standard attention
                # reference_compile=False, # disable triton compile
                )

        def forward(self, input_ids, attention_mask):
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            res = self.model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
            return res
        
    with DisableCompileContextManager():
        tokenizer = AutoTokenizer.from_pretrained("./bert/ruri-v3-30m", model_max_length=4096)

        model = ONNXBert()
        model.eval()

        samples = ['example']

        tokenized = tokenizer(samples,
                return_tensors='pt',
                max_length=tokenizer.model_max_length,
                padding='max_length',
                truncation=True)
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        #model = model
        with torch.no_grad():
            torch.onnx.export(
                    model,
                    (input_ids, attention_mask),
                    onnx_temp_model_path,
                    input_names=["input_ids", "attention_mask"],
                    output_names=["output"],
                    dynamic_axes={
                        "input_ids": {0: "batch_size", 1: "sequence_length"},
                        "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    },
                    )

    #onnx_model = onnx.load(onnx_temp_model_path)
    #simplified_onnx_model, check = simplify(onnx_model)
    """
    fp16_model = float16_converter.convert_float_to_float16(
            simplified_onnx_model,
            keep_io_types=True,  # 入出力は float32 のまま
            disable_shape_infer=True,
        )
    """
    
    #onnx.save(fp16_model, onnx_fp16_model_path)
if __name__ == '__main__':
    export()