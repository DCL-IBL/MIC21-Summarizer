from detectron2 import model_zoo, engine, config
from transformers import AutoModelForCausalLM
import torch
import pdb
import pickle,json,gc,os
import random

import sys
from mic21s import MIC21Summarizer

from datasets import load_dataset
import numpy as np
from tqdm import tqdm

cuda_id = 0
device_map = "auto"

cuda_id = 2
device_map = torch.load('llama_3_dm.pth')
device_map['model.embed_tokens'] = 1
device_map['model.norm'] = 2
device_map['model.rotary_emb'] = 2
device_map['lm_head'] = 1
device_map['model.layers.12'] = 2
device_map['model.layers.13'] = 2
device_map['model.layers.14'] = 2
device_map['model.layers.15'] = 2
device_map['model.layers.16'] = 2
device_map['model.layers.17'] = 2
device_map['model.layers.18'] = 2
device_map['model.layers.19'] = 2
device_map['model.layers.20'] = 2
device_map['model.layers.21'] = 2
device_map['model.layers.22'] = 2
device_map['model.layers.23'] = 2
device_map['model.layers.0'] = 1
device_map['model.layers.1'] = 1
device_map['model.layers.2'] = 2
device_map['model.layers.3'] = 2

model = MIC21Summarizer(cuda_id,device_map)

#dataset = load_dataset("jkralev/mic21")
dataset = load_dataset("mic21_dataset")

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
num_epochs = 10

def load_checkpoint():
    model.projection_layer.load_state_dict(torch.load('model_base_title_1.pth'))
    model.projection_norm.load_state_dict(torch.load('model_base_title_2.pth'))
    return torch.load('model_base_title_loss.pth')

def model_checkpoint():
    torch.save(model.projection_layer.state_dict(), f'model_base_title_1.pth')
    torch.save(model.projection_norm.state_dict(), f'model_base_title_2.pth')
    torch.save(loss_hist,f'model_base_title_loss.pth')

try:
    loss_hist = load_checkpoint()
    last_loss = loss_hist[-1]
    last_iter = len(loss_hist)
    print(f"Checkpoint loaded. Last loss: {last_loss}. Last iter {last_iter}")
except:
    print("No previous checkpoint")
    loss_hist = []

toggle = False
avg_loss = 0
batch_size = 7
seq_len = 40
counter = 0
dataset_subset = dataset["train"]
for ind in tqdm(range(0,dataset_subset.num_rows,batch_size)):
    target_text = dataset_subset[ind:ind+batch_size]["title"]
    target_tok = model.tokenizer(target_text, add_special_tokens=False, max_length=seq_len, padding='max_length')
    
    img_np = [np.array(img) for img in dataset_subset[ind:ind+batch_size]["image"]]
    try:
        img_np = [img[:, :, 2::-1].astype(np.uint8) for img in img_np]       # Convert RGB to BGR
    except Exception as e:
        print(e)
        continue
        
    out1 = model(img_np,seq_len-1)
    loss = torch.nn.CrossEntropyLoss()(out1.permute((0,2,1)), torch.LongTensor(target_tok["input_ids"]).cuda(model.out_device))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_hist.append(loss.item())
    avg_loss += loss.item()
    counter += 1
    if counter % 100 == 0: 
        model_checkpoint()
        decoded_text = model.tokenizer.batch_decode(torch.argmax(out1, dim=-1),skip_special_tokens=True)
        print(decoded_text)
        avg_loss = avg_loss / counter
        print(f"Loss: {avg_loss}")
        avg_loss = 0
        counter = 0