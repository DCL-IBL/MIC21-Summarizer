from transformers import PretrainedConfig
import torch

class MIC21SummarizerConfig(PretrainedConfig):
    model_type = "mic21_summarizer"

    def __init__(
        self,
        hf_text_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        hf_image_model = "microsoft/resnet-50",
        im_model_cuda_id = 0,
        device_map = "auto",
        memory_map = {},
        #text_model_dtype = torch.float16,
        attn_implementation = "eager",
        in_device = 0,
        out_device = 0,
        output_length = 40,
        **kwargs,
    ):
        self.hf_text_model = hf_text_model
        self.hf_image_model = hf_image_model
        self.im_model_cuda_id = im_model_cuda_id
        self.device_map = device_map
        self.memory_map = memory_map
        #self.text_model_dtype = text_model_dtype
        self.attn_implementation = attn_implementation
        self.in_device = in_device
        self.out_device = out_device
        self.output_length = output_length
        self.auto_map = {
                "AutoConfig": "jkralev/mic21_model--configuration_mic21.MIC21SummarizerConfig",
                "AutoModel": "jkralev/mic21_model--modeling_mic21.MIC21SummarizerModel"}
        super().__init__(**kwargs)