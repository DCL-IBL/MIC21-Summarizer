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

cuda_id = 0
device_map = "auto"

model = MIC21Summarizer(cuda_id,device_map)

dataset = load_dataset("jkralev/mic21")

domain_name = "cricket"

indices = [i for i, label in enumerate(dataset['train']['label']) if label == domain_name]
data_subset = dataset['train'].select(indices)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
num_epochs = 1000

toggle = False
for epoch in range(num_epochs):
    for cat in [domain_name]:
        epoch_loss = 0
        for ind in range(data_subset.num_rows):
            target_text = data_subset[ind]["title"]
            target_tok = model.tokenizer(target_text, add_special_tokens=False, max_length=128, padding='max_length')
            
            img_np = np.array(data_subset[ind]["image"])        # Converts to numpy array in RGB order
            img_np = img_np[:, :, ::-1]       # Convert RGB to BGR
            img_np = img_np.astype(np.uint8)  # Ensure dtype is uint8
            
            out1 = model(img_np,128)
            loss = torch.nn.CrossEntropyLoss()(out1.permute((0,2,1)), torch.tensor(target_tok["input_ids"]).unsqueeze(0).cuda(model.out_device))
            #else:
            #    out1 = model(img,target["input_ids"][:,:-1].cuda(cuda_id))
            #    loss = torch.nn.CrossEntropyLoss()(out1.permute((0,2,1)), target["input_ids"][:,1:].cuda(cuda_id))
            #toggle = not toggle
        
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()/data_subset.num_rows
            #if ind > 10:
            #break
        loss_hist[cat].append(epoch_loss)
        decoded_text = model.tokenizer.batch_decode(torch.argmax(out1, dim=-1))
        print(decoded_text)
        print(f"Epoch [{epoch+1}/{num_epochs}], Category: {cat}, Loss: {epoch_loss}")

torch.save(model.projection_layer.state_dict(), f'model_{domain_name}_title_1.pth')
torch.save(model.projection_norm.state_dict(), f'model_{domain_name}_title_2.pth')
torch.save(loss_hist,f'model_{domain_name}_title_loss.pth')