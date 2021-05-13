import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision
from dataloaders.image_caption_dataset import ImageCaptionDataset
from .googlenet_places205 import GoogLeNetPlaces205
from .googlenet_places205_caffe import GoogleNetPlaces205Caffe


class ClassifierScorer():
    def __init__(self, model_path, model_type="googlenet"):
        self.model = None

        if model_type == "googlenet":
            self.model = GoogLeNetPlaces205()
            self.model.load_state_dict(torch.load(model_path))
        elif model_type == "googlenetcaffe":
            self.model = GoogleNetPlaces205Caffe(model_path)
        elif model_type == "googlenetcaffenikhil":
            layer_map = {"conv1_1": "features.0", "conv1_2": "features.2", "conv2_1": "features.5", "conv2_2": "features.7", "conv3_1": "features.10", "conv3_2": "features.12", "conv3_3": "features.14", "conv4_1": "features.17", "conv4_2": "features.19", "conv4_3": "features.21", "conv5_1": "features.24", "conv5_2": "features.26", "conv5_3": "features.28", "fc6": "classifier.0", "fc7": "classifier.3", "fc8": "classifier.6"}
            self.model = torchvision.models.vgg16(num_classes=205)
            s = torch.load(model_path)
            self.model.load_state_dict({self.replace(kn, layer_map):v for kn, v in s.items()})   

        self.model.eval()

    def replace(self, key, mapping):
        k = key[:key.rfind(".")]
        return key.replace(k, mapping[k])

    def score(self, img):
        if self.model is None:
            return None

        return self.model(img).argmax(dim=1).item()