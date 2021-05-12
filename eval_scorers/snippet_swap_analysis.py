import argparse
import numpy as np
import json
import collections
import os
import pickle
import seaborn as sns
import librosa
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd
import simpleaudio as sa

PARTITIONS_DIR = "../partitions"
UTTERANCE_DIR = os.path.join(PARTITIONS_DIR, "test_utterances/")

def read_partition(path):
	data = None
	with open(path, 'r') as f:
		data = json.loads(f.read())

	return data


def load_audio(utterance_dir, utterance_id, sample_rate):
	path = os.path.join(utterance_dir, ("utterance_%s" % utterance_id))
	wav_path = os.path.join(path, ("utterance_%s.wav" % utterance_id))
	audio_ts, _ = librosa.load(wav_path, sample_rate)
	return audio_ts


def load_utterance_metadata(data, utterance_id, speaker_filter = None):
	for d in data:
		uid = d['wav'].split('/')[-1].split(".")[0].split("_")[1]
		if uid == utterance_id:
			if speaker_filter is None or speaker_filter == d['speaker']:
				d['class'] = d['image'].split('/')[1]
				return d
	return None


def plot_audio(audio, sample_rate):
	plt.figure(figsize=(14, 5))
	librosa.display.waveplot(audio, sr=sample_rate)
	plt.show()


def play_audio(audio, sample_rate, bytes_per_sample):
	play_obj = sa.play_buffer(audio, 1, bytes_per_sample, sample_rate)
	play_obj.wait_done()


if __name__ == "__main__":
	paritions = ["train_2020.json", "test_seen_2020.json", "test_unseen_2020.json"]
	parser = argparse.ArgumentParser()
	parser.add_argument("--partitions", type=list, default=[1, 2])
	parser.add_argument("--sample_rate", type=int, default=16000)

	args = parser.parse_args()


	data = []
	for p in args.partitions:
		data += read_partition(os.path.join(PARTITIONS_DIR, paritions[p]))['data']
	

	sr = args.sample_rate
	bytes_per_sample = 4

	utterances = {}
	for d in os.listdir(UTTERANCE_DIR):
		utterance_id = d.split("_")[1]
		audio = load_audio(UTTERANCE_DIR, utterance_id, sr)
		meta = load_utterance_metadata(data, utterance_id)
		if meta is None:
			print("Could not find metadata for utterance %s" % utterance_id)
			continue

		utterances[utterance_id] = meta
		utterances[utterance_id]['audio'] = audio
		# play_audio(audio, sr, bytes_per_sample)

	
	for u in utterances.values():
		print(u['speaker'])

	

# librosa.output.write_wav('audio/tone_440.wav', x, sr)
