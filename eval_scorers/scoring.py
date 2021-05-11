import math
import sys
import time
import os.path
import numpy as np
import scipy.signal
import scipy.misc
import librosa
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import dave_models
import matplotlib.pyplot as plt
import tarfile

from PIL import Image
from googlenet_places205 import GoogLeNetPlaces205
from dataloaders.image_caption_dataset import ImageCaptionDataset


GOOGLENET_PLACES205_PATH = 'eval_scorers/trained_models/googlenet_places205/googlenet_places205.pth' 
DAVENET_MODEL_PATH = 'eval_scorers/trained_models/davenet_vgg16_MISA_1024_pretrained/'
AUDIO_MODEL_PATH = os.path.join(DAVENET_MODEL_PATH, 'audio_model.pth')
IMAGE_MODEL_PATH = os.path.join(DAVENET_MODEL_PATH, 'image_model.pth')

TEST_DATA_DIR = "./data/"
DATASET_BASE_PATH = os.path.join(TEST_DATA_DIR, "PlacesAudioEnglish")
TAR_FILE_PATH = DATASET_BASE_PATH + ".tar.gz"


class DaveNetScorer():
    def __init__(self, audio_model_path, image_model_path, matchmap_thresh = 5.0):
        self.audio_model, self.image_model = dave_models.DAVEnet_model_loader(audio_model_path, image_model_path)
        self.audio_model.eval()
        self.image_model.eval()
        self.image_transform = transforms.Compose(
                [
                transforms.ToPILImage(mode='RGB'),
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

        self.matchmap_thresh = matchmap_thresh


    def get_image_features(self, img):
        with torch.no_grad():
            self.image_model.eval()
            image_transformed = self.image_transform(img).unsqueeze(0)
            image_feature_map = self.image_model(image_transformed).squeeze(0)
            emb_dim = image_feature_map.size(0)
            output_H = image_feature_map.size(1)
            output_W = image_feature_map.size(2)
            return image_feature_map.view(emb_dim, output_H * output_W), (emb_dim, output_H, output_W)


    def get_audio_features(self, melspec):
        with torch.no_grad():
            audio_output = self.audio_model(melspec.unsqueeze(0).unsqueeze(0)).squeeze(0)
            return audio_output


    def score(self, melspec, img):
        # Scores produced using metrics defined by https://arxiv.org/pdf/1804.01452.pdf
        image_output, image_dim = self.get_image_features(img)
        audio_output = self.get_audio_features(melspec)
        _, img_output_H, img_output_W = image_dim
        heatmap = torch.mm(audio_output.t(), image_output).squeeze()
        heatmap = heatmap.view(audio_output.size(1), img_output_H, img_output_W).numpy()#.max(dim=0)[0].numpy()
        print(heatmap)
        matches = np.where(heatmap >= self.matchmap_thresh, 0, 1)
        N_t = audio_output.size(1)
        N_r = img_output_H
        N_c = img_output_W
        sisa = np.sum(heatmap) / (N_t * N_r * N_c)
        misa = np.sum(np.max(heatmap.reshape(N_t, N_r * N_c), axis = 1)) / (N_t)
        sima = np.sum(np.max(heatmap, axis = 0)) / (N_r * N_c)
        return heatmap, matches, sisa, misa, sima   


class ClassifierScorer():
    def __init__(self, model_path, model_type = "GoogLeNetPlaces205"):
        self.model = None

        if model_type == "GoogLeNetPlaces205":
            self.model = GoogLeNetPlaces205()
            self.model.load_state_dict(torch.load(model_path))
            
        self.model.eval()


    def score(self, img):
        if self.model is None:
            return None

        return self.model(img) 


def main():
    dave_scorer = DaveNetScorer(AUDIO_MODEL_PATH, IMAGE_MODEL_PATH)
    clf_scorer = ClassifierScorer(GOOGLENET_PLACES205_PATH)

    if os.path.isfile(TAR_FILE_PATH):
          tar = tarfile.open(TAR_FILE_PATH, "r:gz")
          tar.extractall(path=TEST_DATA_DIR)

    loader = ImageCaptionDataset(os.path.join(DATASET_BASE_PATH, "samples.json"))

    for img, audio, n_frames in loader:
        heatmap, matches, sisa, misa, sima  = dave_scorer.score(audio, img)
        clf_score = clf_scorer.score(img.unsqueeze(0))
        print(clf_score.shape)
        print("SISA", sisa)
        print("MISA", misa)
        print("SIMA", sima)
        print("CLF", clf_score)
        fig, ax = plt.subplots(1, 2, figsize=(25, 5), gridspec_kw={'width_ratios': [1, 3]})
        ax[0].imshow(img.permute(1, 2, 0))
        ax[1].imshow(audio, aspect='auto')
        plt.show()
        

if __name__ == "__main__":
    main()