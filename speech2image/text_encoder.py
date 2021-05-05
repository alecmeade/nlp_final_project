import torch
import numpy
from torch.utils.data import DataLoader, TensorDataset
from transformers import *


class TextEncoder(torch.nn.Module):
    def __init__(self, train=False):
        self.tokenizer = T5Tokenizer.from_pretrained("t5-large")
        self.model = T5EncoderModel.from_pretrained("t5-large", return_dict=True)
        self.model.train() if train else self.model.eval()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def forward(self, data):
        batch_size = data.shape[0]
        ids = self.tokenizer(data, padding="max_length", max_length=512, truncation=True, return_tensors="pt")["input_ids"]
        train_data = TensorDataset(ids)
        train_dataloader = DataLoader(train_data, batch_size=batch_size)
        
        embedding = []
        start = time.time()
        
        with torch.no_grad():
            for step_num, batch in enumerate(train_dataloader):
                description = batch[0]
                description = description.to(self.device)
            
            output = self.model(description)
            embedding.append(output.last_hidden_state)
        
        embed = torch.cat(embedding, dim=1)
        seq_embed = torch.sum(embed, dim=1)
        return seq_embed