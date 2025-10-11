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

model = MIC21Summarizer(cuda_id,device_map)

dataset = load_dataset("jkralev/mic21")

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
num_epochs = 10

loss_hist = []

def model_checkpoint():
    torch.save(model.projection_layer.state_dict(), f'model_base_title_1.pth')
    torch.save(model.projection_norm.state_dict(), f'model_base_title_2.pth')
    torch.save(loss_hist,f'model_base_title_loss.pth')

toggle = False
avg_loss = 0
for ind in tqdm(range(dataset["train"].num_rows)):
    target_text = dataset["train"][ind]["title"]
    target_tok = model.tokenizer(target_text, add_special_tokens=False, max_length=64, padding='max_length')
            
    img_np = np.array(dataset["train"][ind]["image"])        # Converts to numpy array in RGB order
    img_np = img_np[:, :, ::-1]       # Convert RGB to BGR
    img_np = img_np.astype(np.uint8)  # Ensure dtype is uint8
            
    out1 = model(img_np,64)
    loss = torch.nn.CrossEntropyLoss()(out1.permute((0,2,1)), torch.tensor(target_tok["input_ids"]).unsqueeze(0).cuda(model.out_device))
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_hist.append(loss.item())
    avg_loss += loss.item()
    if ind % 100 == 0: 
        decoded_text = model.tokenizer.batch_decode(torch.argmax(out1, dim=-1),skip_special_tokens=True)
        print(decoded_text)
        avg_loss = avg_loss / 100
        print(f"Epoch {avg_loss}")
        model_checkpoint()