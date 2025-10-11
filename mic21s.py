from detectron2 import model_zoo, engine, config
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from detectron2.data.detection_utils import read_image
import torch
import pdb
import pickle,json,gc,os
import random

import matplotlib.pyplot as plt

from transformers import OffloadedCache

class MIC21Summarizer(torch.nn.Module):
    def __init__(self,cuda_id,device_map):
        super().__init__()
        self.cuda_id = cuda_id
        #self.llm_name = "meta-llama/Llama-3.1-8B-instruct"
        self.llm_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.device_map = device_map
        #Init Image Processing Model
        self.cfg = config.get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'))
        self.cfg.MODEL.WEIGHTS = 'detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl'
        self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.9
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
        self.cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.9
        self.cfg.MODEL.DEVICE = f"cuda:{self.cuda_id}"
        
        self.img_predictor = engine.DefaultPredictor(self.cfg)

        #self.quantization_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16)
        #self.llm = AutoModelForCausalLM.from_pretrained(
        #    self.llm_name,
        #    torch_dtype=torch.float16,
        #    device_map=device_map,
        #    attn_implementation="eager",
        #    quantization_config=self.quantization_config)
        #self.llm.gradient_checkpointing_enable()
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.llm_name,
            device_map="auto",
            #max_memory={1: "5GiB",2: "5GiB",},
            torch_dtype=torch.float16,
            attn_implementation="eager"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name)

        self.in_device = 0 #device_map['model.embed_tokens']
        self.out_device = 0 #device_map['lm_head']

        self.projection_layer = torch.nn.Linear(256, self.llm.config.hidden_size, dtype=torch.float, device=f"cuda:{self.in_device}")
        self.projection_norm = torch.nn.LayerNorm(256, eps=1e-5, bias=True, device=f"cuda:{self.in_device}")
        self.projection_dropout = torch.nn.Dropout(0.1)

        for param in self.img_predictor.model.parameters():
            param.requires_grad = False

        for param in self.llm.parameters():
            param.requires_grad = False

    def get_img_features(self,img):
        if self.img_predictor.input_format == "RGB":
            img = img[:, :, ::-1]
        height, width = img.shape[:2]
        image = self.img_predictor.aug.get_transform(img).apply_image(img)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        #image.to(self.img_predictor.cfg.MODEL.DEVICE)
        image.cuda(self.cuda_id)
        inputs = {"image": image, "height": height, "width": width}
        #predictions = predictor.model.forward([inputs])
        images = self.img_predictor.model.preprocess_image([inputs])
        features = self.img_predictor.model.backbone(images.tensor)
        pooled = torch.nn.AdaptiveAvgPool2d((16,16))(features['p6'])
        return pooled.view(1,256,256)
        
    def forward(self, img, target_len):
        img_features = self.get_img_features(img)

        messages = [
            {"role":"system","content":"Generate title and description for the provided image. The image features are: "},
            {"role":"user","content":"Generate a title:"}
                        ]
        tokenized_messages = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.in_device)
        vectorized_messages = self.llm.model.embed_tokens(tokenized_messages[0]).unsqueeze(0)
        first_eos_index = (tokenized_messages[0]==2).nonzero()[0].item()

        visual_embeddings = self.projection_layer(self.projection_dropout(self.projection_norm(img_features.to(f"cuda:{self.in_device}"))))
        #combined_embeds = torch.cat([self.msg_p1_emb, visual_embeddings.half().to(self.in_device), self.msg_p2_emb],dim=1)
        combined_embeds = torch.cat([
            vectorized_messages[:,:first_eos_index-1,:], 
            visual_embeddings.half().to(self.in_device), 
            vectorized_messages[:,first_eos_index:,:]],dim=1)

        #combined_embeds = torch.cat([self.input_emb, self.eot_emb],dim=1)
        self.cache = OffloadedCache()
        
        outputs = self.llm(inputs_embeds=combined_embeds,past_key_values=self.cache,use_cache=True)
        logits = outputs.logits[:,-1]
        out_logits = None
        last_position_id = combined_embeds.shape[1] - 1
        new_tok = torch.argmax(logits,dim=-1)

        if target_len is None:
            max_len = 64
        else:
            max_len = target_len
        for k in range(0,max_len):
            position_ids = torch.tensor([[last_position_id + 1]], device=combined_embeds.device)
            last_position_id += 1
            outputs = self.llm(input_ids=new_tok.unsqueeze(0), 
                               past_key_values=self.cache,
                               use_cache=True,
                               position_ids=position_ids)
            logits = outputs.logits[:,-1]
            if out_logits is None:
                out_logits = logits.unsqueeze(0)
            else:
                out_logits = torch.cat([out_logits,logits.unsqueeze(0)],dim=1)
            new_tok = torch.argmax(logits,dim=-1)
            if target_len is None and new_tok.item() == self.tokenizer.eos_token_id:
                break
        
        return out_logits

    def generate_causal_mask(self,size):
        #return torch.block_diag(torch.zeros(size, size), torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)).cuda(cuda_id)
        return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1).cuda(cuda_id)