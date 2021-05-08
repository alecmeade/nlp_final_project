# ImageCaptation Places and Text Dataloaders implementation borrowed from
# https://github.com/dharwath/DAVEnet-pytorch/blob/master/dataloaders/image_caption_dataset.py
# which is referenced on the CSAIL website https://groups.csail.mit.edu/sls/downloads/placesaudio/downloads.cgi.

import json
import librosa
import numpy as np
import os
from PIL import Image
import scipy.signal
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageCaptionTextDataset(Dataset):
    def __init__(self, dataset_json_file, image_conf=None, img_size=256):
        """
        Dataset that manages a set of paired images and audio recordings
        :param dataset_json_file
        :param audio_conf: Dictionary containing the sample rate, window and
        the window length/stride in seconds, and normalization to perform (optional)
        :param image_transform: torchvision transform to apply to the images (optional)
        """
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        self.data = data_json['data']
        self.image_base_path = data_json['image_base_path']

        if not image_conf:
            self.image_conf = {}
        else:
            self.image_conf = image_conf

        crop_size = self.image_conf.get('crop_size', img_size)
        center_crop = self.image_conf.get('center_crop', False)

        if center_crop:
            self.image_resize_and_crop = transforms.Compose(
                [transforms.Resize(img_size), transforms.CenterCrop(img_size), transforms.ToTensor()])
        else:
            self.image_resize_and_crop = transforms.Compose(
                [transforms.RandomResizedCrop(crop_size), transforms.ToTensor()])

        RGB_mean = self.image_conf.get('RGB_mean', [0.485, 0.456, 0.406])
        RGB_std = self.image_conf.get('RGB_std', [0.229, 0.224, 0.225])
        self.image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

    def _LoadText(self, d):
        return d["asr_text"]

    def _LoadImage(self, impath):
        img = Image.open(impath).convert('RGB')
        img = self.image_resize_and_crop(img)
        # img = self.image_normalize(img)
        return img

    def __getitem__(self, index):
        datum = self.data[index]
        imgpath = os.path.join(self.image_base_path, datum['image'])
        text = self._LoadText(datum)
        image = self._LoadImage(imgpath)
        return image, text

    def __len__(self):
        return len(self.data)
