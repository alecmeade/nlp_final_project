import time
import torch
import numpy
from torch.utils.data import DataLoader, TensorDataset
from transformers import T5Tokenizer, T5EncoderModel


class TextEncoder(torch.nn.Module):
    def __init__(self, train=True):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model = T5EncoderModel.from_pretrained("t5-small", return_dict=True).to(self.device)
        self.model.train() if train else self.model.eval()
    
    def forward(self, data):
        ids = self.tokenizer(data, padding="max_length", max_length=512, truncation=True, return_tensors="pt")["input_ids"].to(self.device)
        seq_embed = self.model(ids).last_hidden_state.mean(dim=1)
        return seq_embed