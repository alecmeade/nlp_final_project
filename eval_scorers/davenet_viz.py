import os
import argparse
import json
import torch
import librosa.display
import torchvision.transforms as transforms
import simpleaudio as sa
from dataloaders.image_caption_dataset import ImageCaptionDataset
from davenet_scorer import DaveNetScorer
import cv2
import numpy as np
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import scipy
from PIL import Image

from textwrap import wrap


DAVENET_PATH = "eval_scorers/trained_models/davenet_vgg16_MISA_1024_pretrained/"
DAVENET_AUDIO = os.path.join(DAVENET_PATH, "audio_model.pth")
DAVENET_IMAGE = os.path.join(DAVENET_PATH, "image_model.pth")

def play_audio(audio, sample_rate, bytes_per_sample, wait=True):
    play_obj = sa.play_buffer(audio, 1, bytes_per_sample, sample_rate)
    if wait:
        play_obj.wait_done()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/PlacesAudioEnglish/samples.json", help="Path to the dataset.")
    parser.add_argument("--model_dir", type=str, default="./eval_scorers/trained_models", help="Path to pretrained models.")
    parser.add_argument("--audio_model", type=str, default=DAVENET_AUDIO, help="Path to the DaveNet audio model.")
    parser.add_argument("--image_model", type=str, default=DAVENET_IMAGE , help="Path to the DaveNet image model.")
    parser.add_argument("--window_size", type=int, default=7, help="Size of the temporal window to conduct smoothing of potential matches.")
    parser.add_argument("--smoothing", type=str, default="max", help="Either max or mean")
    parser.add_argument("--mass_p", type=float, default=0.3, help="The percentage of image mass to retain within a matchup when masking the image.")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate.")
    parser.add_argument("--device", type=str, default="cpu", help="Either cuda or cpu.")
    parser.add_argument("--outdir", type=str, default="./scores", help="Output dir for plots, reports, etc.")
    args = parser.parse_args()
          
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    smoothing_fn = None
    if args.smoothing == "max":
        smoothing_fn = np.max
    elif args.smoothing == "mean":
        smoothing_fn = np.mean
    else:
        print("Defaulting to max smoothing...")
        smoothing_fn = np.max

    # Load DaveNet
    dave_scorer = DaveNetScorer(args.audio_model, args.image_model)
   
    # Dataset
    audio_conf = {"use_raw_length": False}
    dataset = ImageCaptionDataset(args.dataset, audio_conf=audio_conf, normalize=False)
    data_key = {x["wav"]:x for x in dataset.data}
    loader = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=1)
        
    for img, audio, p, wavpath in loader:
        l = data_key[wavpath[0][wavpath[0].find("wavs"):]]
        uid = l["uttid"]
        asr_text = l["asr_text"]
 
        wav, _ = librosa.load(wavpath[0], args.sample_rate)
        # play_audio(wav, args.sample_rate, 4)
        heatmap, _, _, _, _ = dave_scorer.score(audio.squeeze(0), img.squeeze(0))
        N_t, N_r, N_c = heatmap.shape

        temporal_heatmap = heatmap.reshape((N_t, N_r * N_c))
        smoothed_heatmap = np.zeros(temporal_heatmap.shape)

        for i in range(0, N_t - args.window_size):
          smoothed_heatmap[i, :] = np.max(temporal_heatmap[i: i + args.window_size, :], axis = 0)
        
        # Normalize Matches
        total_mass = np.sum(smoothed_heatmap)
        matchmap = smoothed_heatmap / total_mass


        # Sort cells
        N = N_t * N_r * N_c
        matchmap = matchmap.reshape(N)
        sorted_cells = np.argsort(matchmap)

        # Retain only cells accounting for mass_p percent of the total density in the image
        mass_thresh = args.mass_p * np.sum(matchmap)
        sum_mass = 0.0
        for i in range(N - 1, -1, -1):
            idx = sorted_cells[i]
            if (matchmap[idx] + sum_mass) <= mass_thresh:
                sum_mass += matchmap[idx]

            else:
                matchmap[idx] = 0.0


        matchmap = matchmap.reshape((N_t, N_r, N_c))
        img_t = np.transpose(img.squeeze(0).numpy(), [1, 2, 0])
        wav_duration = librosa.get_duration(wav, sr=args.sample_rate)
        time_step = wav_duration / (N_t)
        max_val = np.max(matchmap)

        for i in range(0, (N_t - args.window_size)):
            f, ax = plt.subplots(3, 1)
            title = "\n".join(wrap("%s: %s" % (uid, asr_text), 40))
            f.suptitle(title)
            ax[0].imshow(img_t, aspect='auto')

            # Filter to matchmap at this time step
            full_matchmap = np.array(Image.fromarray(matchmap[i, :, :]).resize((img_t.shape[0], img_t.shape[1])))

            # Trying to remove vmin and vmax helps make it clearer to see changes through time.
            ax[1].imshow(full_matchmap, aspect='auto', vmin=0.0, vmax=max_val)
            ax[0].get_shared_x_axes().join(ax[0], ax[1])
            librosa.display.waveplot(wav, sr=args.sample_rate, ax=ax[2])
            ax[2].axvline(i * time_step, linewidth=2, color='k', linestyle='dashed')
            plt.savefig(args.outdir + "/%s_%d.jpg" % (uid, i))
            plt.close()
        

if __name__ == "__main__":
    main()

    
    