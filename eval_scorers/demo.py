import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import pytorch_lightning as pl
import sounddevice as sd
from speech2image.model import Speech2Image


def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    outdata[:] = indata

if __name__ == "__main__":
	model_dir = "trained_models/"
	speech_model_path = os.path.join(model_dir, "stylegan2-050721_2039/speech2image_epoch=0500.ckpt")
	speech_model = Speech2Image()
	speech_model.load_state_dict(torch.load(speech_model_path, map_location=torch.device('cpu')))
	speech_model.eval()

	sr = 16000
	duration_sec = 5
	while True:
		
		audio = sd.rec(int(duration_sec * sr), samplerate=sr, channels=1)
		sd.wait()
		sd.play(audio, sr)
		sd.wait()
		img = speech_model(audio)
		plt.figure()
		plt.imshow(img)
		plt.show()