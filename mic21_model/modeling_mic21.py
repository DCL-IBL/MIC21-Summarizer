from transformers.modeling_utils import PreTrainedModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pdb
from transformers import OffloadedCache,DynamicCache
from .configuration_mic21 import MIC21SummarizerConfig
import numpy as np
from transformers import AutoImageProcessor, ResNetForImageClassification

class MIC21SummarizerModel(PreTrainedModel):
    config_class = MIC21SummarizerConfig
    is_parallelizable = True
    model_parallel = True
    place_model_on_device = False
    model_wrapped = {}

    def init_components(self):
        self.components["image_model"] = ResNetForImageClassification.from_pretrained(self.hf_config.hf_image_model).cuda()
        self.components["image_processor"] = AutoImageProcessor.from_pretrained(self.hf_config.hf_image_model)
        
        self.components["llm"] = AutoModelForCausalLM.from_pretrained(self.hf_config.hf_text_model,torch_dtype=torch.float16).cuda()
        self.components["tokenizer"] = AutoTokenizer.from_pretrained(self.hf_config.hf_text_model)

        for param in self.components["image_model"].parameters():
            param.requires_grad = False

        for param in self.components["llm"].parameters():
            param.requires_grad = False
    
    def __init__(self,config):
        super().__init__(config)
        #Init Image Processing Model        
        self.components = {"image_model":None,"llm":None,"tokenizer":None,"image_processor":None}
        self.hf_config = config
        #self.components["image_model"] = ResNetForImageClassification.from_pretrained(config.hf_image_model,device_map=f"cuda:{config.im_model_cuda_id}")
        #self.components["image_model"] = ResNetForImageClassification.from_pretrained(config.hf_image_model).cpu().cuda()
        
        #self.components["image_processor"] = AutoImageProcessor.from_pretrained(config.hf_image_model)

        #self.components["llm"] = AutoModelForCausalLM.from_pretrained(config.hf_text_model,torch_dtype=torch.float16).cpu().cuda()
        
        #self.quantization_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16)
        #self.components["llm"] = AutoModelForCausalLM.from_pretrained(
        #    config.hf_text_model,
        #    device_map=config.device_map,
        #    max_memory=config.memory_map,
        #    torch_dtype=torch.float16,#config.text_model_dtype,
        #    attn_implementation=config.attn_implementation,
        #    #quantization_config=self.quantization_config
        #)
        #self.components["tokenizer"] = AutoTokenizer.from_pretrained(config.hf_text_model)

        #self.in_device = config.in_device
        #self.out_device = config.out_device

        #self.projection_layer = torch.nn.Linear(49, self.components["llm"].config.hidden_size, dtype=torch.float, device=f"cuda:{self.in_device}")
        self.projection_layer = torch.nn.Linear(49, 2048, dtype=torch.float)
        
        #self.projection_norm = torch.nn.LayerNorm(49, eps=1e-5, bias=True, device=f"cuda:{self.in_device}")
        self.projection_norm = torch.nn.LayerNorm(49, eps=1e-5, bias=True)
        self.projection_dropout = torch.nn.Dropout(0.1)

        self.im_model_cuda_id = config.im_model_cuda_id
        self.output_length = config.output_length
        
    def forward(self, images, titles):
        prepared_images = self.components["image_processor"](images,return_tensors="pt")
        prepared_images["pixel_values"] = prepared_images["pixel_values"].cuda()
        #prepared_images = prepared_images.to(f"cuda:{self.im_model_cuda_id}")
        
        img_features = self.components["image_model"](**prepared_images,output_hidden_states=True)
        img_features = img_features["hidden_states"][-1]
        (batch_size,nfilter,nx,ny)=img_features.shape
        img_features = img_features.view(batch_size,nfilter,nx*ny)

        messages = [
            {"role":"system","content":"Generate title and description for the provided image. The image features are: "},
            {"role":"user","content":"Generate a title:"}]
        
        tokenized_messages = self.components["tokenizer"].apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
            #.to(self.in_device)     
        vectorized_messages = self.components["llm"].model.embed_tokens(tokenized_messages[0]).unsqueeze(0)
        vectorized_messages = vectorized_messages.repeat(batch_size,1,1)
            #.to(self.in_device)
        first_eos_index = (tokenized_messages[0]==self.components["tokenizer"].eos_token_id).nonzero()[0].item()

        #img_features = img_features.to(f"cuda:{self.in_device}")
        visual_embeddings = self.projection_layer(self.projection_dropout(self.projection_norm(img_features[:,0:256,:])))

        #visual_embeddings.half().to(self.in_device)
        combined_embeds = torch.cat([
            vectorized_messages[:,:first_eos_index-1,:], 
            visual_embeddings.half(), 
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
            if max_len is None and new_tok.item() == self.components["tokenizer"].eos_token_id:
                break
        if titles is not None:
            target_tok = self.components["tokenizer"](titles, add_special_tokens=False, max_length=max_len+1, padding='max_length')
            loss = torch.nn.CrossEntropyLoss()(out_logits.permute((0,2,1)), torch.LongTensor(target_tok["input_ids"]).cuda())
                #.cuda(self.out_device))
            return {"loss": loss, "logits": logits, "eval_loss": loss}
            
        return {"logits":out_logits}
        