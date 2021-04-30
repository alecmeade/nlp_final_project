import torch
import numpy as np

def collate_wav(batch):
    image_inputs = torch.stack([s[0] for s in batch])
    audio_inputs = [torch.from_numpy(s[1]) for s in batch]
    nframes = torch.tensor([s[2] for s in batch])

    max_length = max(nframes)
    
    audio_padded = []
    for sample in audio_inputs:
        audio_padded.append(torch.nn.functional.pad(sample, 
                                   (0, max_length - sample.shape[0]),
                                    "constant",
                                    0))
    
    return image_inputs, torch.stack(audio_padded), nframes
