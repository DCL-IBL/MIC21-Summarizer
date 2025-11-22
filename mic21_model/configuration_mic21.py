from transformers import PretrainedConfig
import torch

class MIC21SummarizerConfig(PretrainedConfig):
    model_type = "mic21_summarizer"

    def __init__(
        self,
        hf_text_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        detectron2_config_file = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
        detectron2_weights_file = "detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl",
        detectron2_cuda_id = 0,
        device_map = "auto",
        memory_map = {},
        torch_dtype = torch.float16,
        attn_implementation = "eager",
        in_device = 0,
        out_device = 0,
        output_length = 40,
        **kwargs,
    ):
        self.hf_text_model = hf_text_model
        self.detectron2_config_file = detectron2_config_file
        self.detectron2_weights_file = detectron2_weights_file
        self.detectron2_cuda_id = detectron2_cuda_id
        self.device_map = device_map
        self.memory_map = memory_map
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.in_device = in_device
        self.out_device = out_device
        self.output_length = output_length
        super().__init__(**kwargs)