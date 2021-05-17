# Some DaveNet scoring code is borrowed from from https://github.com/iapalm/davenet_demo/tree/main/models
import numpy as np
import torch
import torchvision.transforms as transforms
import dave_models

class DaveNetScorer():
    def __init__(self, audio_model_path, image_model_path, matchmap_thresh = 5.0):
        self.audio_model, self.image_model = dave_models.DAVEnet_model_loader(audio_model_path, image_model_path)
        self.audio_model.eval()
        self.image_model.eval()
        self.image_transform = transforms.Compose(
                [
                transforms.ToPILImage(mode='RGB'),
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

        matches = np.where(heatmap >= self.matchmap_thresh, 0, 1)
        N_t = audio_output.size(1)
        N_r = img_output_H
        N_c = img_output_W
        sisa = np.sum(heatmap) / (N_t * N_r * N_c)
        misa = np.sum(np.max(heatmap.reshape(N_t, N_r * N_c), axis = 1)) / (N_t)
        sima = np.sum(np.max(heatmap, axis = 0)) / (N_r * N_c)
        return heatmap, matches, sisa, misa, sima
        