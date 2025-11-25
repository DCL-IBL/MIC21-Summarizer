from transformers.modeling_utils import PreTrainedModel

import detectron2
from detectron2 import model_zoo,engine
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from detectron2.data.detection_utils import read_image
import torch
import pdb
import pickle,json,gc,os
import random

import matplotlib.pyplot as plt

from transformers import OffloadedCache,DynamicCache

from .configuration_mic21 import MIC21SummarizerConfig

import numpy as np

class MIC21SummarizerModel(PreTrainedModel):
    config_class = MIC21SummarizerConfig
    
    def __init__(self,config):
        super().__init__(config)
        #Init Image Processing Model
        self.detectron2_cfg = detectron2.config.get_cfg()
        self.detectron2_cfg.merge_from_file(model_zoo.get_config_file(config.detectron2_config_file))
        self.detectron2_cfg.MODEL.WEIGHTS = config.detectron2_weights_file
        self.detectron2_cfg.MODEL.DEVICE = f"cuda:{config.detectron2_cuda_id}"

        self.components = {"img_predictor":{},"llm":{},"tokenizer":{}}
        self.components["img_predictor"] = engine.DefaultPredictor(self.detectron2_cfg)

        #self.quantization_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16)
        self.components["llm"] = AutoModelForCausalLM.from_pretrained(
            config.hf_text_model,
            device_map=config.device_map,
            max_memory=config.memory_map,
            torch_dtype=config.torch_dtype,
            attn_implementation=config.attn_implementation,
            #quantization_config=self.quantization_config
        )
        self.components["tokenizer"] = AutoTokenizer.from_pretrained(config.hf_text_model)

        self.in_device = config.in_device
        self.out_device = config.out_device

        self.projection_layer = torch.nn.Linear(256, self.components["llm"].config.hidden_size, dtype=torch.float, device=f"cuda:{self.in_device}")
        self.projection_norm = torch.nn.LayerNorm(256, eps=1e-5, bias=True, device=f"cuda:{self.in_device}")
        self.projection_dropout = torch.nn.Dropout(0.1)

        for param in self.components["img_predictor"].model.parameters():
            param.requires_grad = False

        for param in self.components["llm"].parameters():
            param.requires_grad = False

        self.detectron2_cuda_id = config.detectron2_cuda_id
        self.output_length = config.output_length

    def get_img_features(self,img_array):
        inputs = []
        for img in img_array:
            height, width = img.shape[:2]
            image = self.components["img_predictor"].aug.get_transform(img).apply_image(img)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image.cuda(self.detectron2_cuda_id)
            inputs.append({"image": image, "height": height, "width": width})
        images = self.components["img_predictor"].model.preprocess_image(inputs)
        features = self.components["img_predictor"].model.backbone(images.tensor)
        pooled = torch.nn.AdaptiveAvgPool2d((16,16))(features['p6'])
        batch_size = pooled.shape[0]
        return pooled.view(batch_size,256,256)
        
    def forward(self, images, titles):
        img_np = [np.array(img) for img in images]
        img_np = [img[:, :, 2::-1].astype(np.uint8) for img in img_np]       # Convert RGB to BGR
        
        img_features = self.get_img_features(img_np)
        batch_size = len(img_np)

        messages = [
            {"role":"system","content":"Generate title and description for the provided image. The image features are: "},
            {"role":"user","content":"Generate a title:"}]
        
        tokenized_messages = self.components["tokenizer"].apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.in_device)     
        vectorized_messages = self.components["llm"].model.embed_tokens(tokenized_messages[0]).unsqueeze(0)
        vectorized_messages = vectorized_messages.repeat(batch_size,1,1).to(self.in_device)
        first_eos_index = (tokenized_messages[0]==self.components["tokenizer"].eos_token_id).nonzero()[0].item()

        visual_embeddings = self.projection_layer(self.projection_dropout(self.projection_norm(img_features.to(f"cuda:{self.in_device}"))))
        
        combined_embeds = torch.cat([
            vectorized_messages[:,:first_eos_index-1,:], 
            visual_embeddings.half().to(self.in_device), 
            vectorized_messages[:,first_eos_index:,:]],dim=1)

        #combined_embeds = torch.cat([self.input_emb, self.eot_emb],dim=1)
        self.cache = OffloadedCache()
        #self.cache = DynamicCache()
        
        outputs = self.components["llm"](inputs_embeds=combined_embeds,past_key_values=self.cache,use_cache=True)
        logits = outputs.logits[:,-1]
        out_logits = logits.unsqueeze(1)
        new_tok = torch.argmax(logits,dim=-1)

        if self.output_length is None:
            max_len = 64
        else:
            max_len = self.output_length
            
        for k in range(0,max_len):
            outputs = self.components["llm"](input_ids=new_tok.unsqueeze(0).permute(1,0),past_key_values=self.cache,use_cache=True)
            logits = outputs.logits[:,-1]
            if out_logits is None:
                out_logits = logits.unsqueeze(1)
            else:
                out_logits = torch.cat([out_logits,logits.unsqueeze(1)],dim=1)
            new_tok = torch.argmax(logits,dim=-1)
            if target_len is None and new_tok.item() == self.components["tokenizer"].eos_token_id:
                break
        if targets is not None:
            target_tok = model.tokenizer(titles, add_special_tokens=False, max_length=max_len, padding='max_length')
            loss = torch.nn.CrossEntropyLoss()(out_logits.permute((0,2,1)), torch.LongTensor(target_tok).cuda(self.out_device))
            return {"loss": loss, "logits": logits}
            
        return {"logits":out_logits}
        