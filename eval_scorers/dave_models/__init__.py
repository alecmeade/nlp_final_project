import torch
from .AudioModels import *
from .ImageModels import *
from collections import OrderedDict

def DAVEnet_model_loader(audio_path, image_path):
    audio_model = Davenet()
    image_model = VGG16()
    audio_state_dict = torch.load(audio_path, map_location='cpu')
    image_state_dict = torch.load(image_path, map_location='cpu')
    modified_audio_state_dict = OrderedDict()
    modified_image_state_dict = OrderedDict()

    for k, v in audio_state_dict.items():
        if k.startswith('module'):
            name = k[7:]
        else:
            name = k
        modified_audio_state_dict[name] = v
    audio_pretrained_dict = modified_audio_state_dict

    for k, v in image_state_dict.items():
        if k.startswith('module'):
            name = k[7:]
        else:
            name = k
        modified_image_state_dict[name] = v
    image_pretrained_dict = modified_image_state_dict

    audio_model_dict = audio_model.state_dict()
    image_model_dict = image_model.state_dict()

    audio_pretrained_dict = {k: v for k, v in audio_pretrained_dict.items() if k in audio_model_dict}
    audio_model_dict.update(audio_pretrained_dict)
    audio_model.load_state_dict(audio_model_dict)

    image_pretrained_dict = {k: v for k, v in image_pretrained_dict.items() if k in image_model_dict}
    image_model_dict.update(image_pretrained_dict)
    image_model.load_state_dict(image_model_dict) 

    return audio_model, image_model
