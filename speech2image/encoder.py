import torch
from speech2image.speechbrain_crdnn import CRDNNEncoder

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

encoders = {
        "CRDNN": CRDNNEncoder.from_pretrained
    }

supported_encoders = list(encoders.keys())

def parse_encoder(encoder_str):
    encoder_str = encoder_str.lower()
    if encoder_str not in encoders:
        raise ValueError("unsupported encoder {}".format(encoder_str))
    else:
        return encoders[encoder_str]
