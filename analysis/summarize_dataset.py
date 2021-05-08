import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
import collections
import os
import pickle
import seaborn as sns


def summarize_partition(path, data):
	print("Partition: %s" % path)

	caption_lengths = []
	speaker_counts = collections.defaultdict(int)
	word_counts = collections.defaultdict(int)

	for s in data:
		speaker = s['speaker']
		caption = s['asr_text']


		caption_lengths.append(len(caption.split(" ")))
		speaker_counts[speaker] += 1
		for w in caption.split():
			word_counts[w] += 1

	return caption_lengths, speaker_counts, word_counts


def read_partition(path):
	data = None
	with open(path, 'r') as f:
		data = json.loads(f.read())
		data = data['data']

	return data


def count_hist(cnts, xlabel):
	values = sorted(cnts)
	plt.figure()
	sns.histplot(values)
	plt.xlabel(xlabel)
	plt.ylabel("Counts")
	plt.show()


def frequency_hist(cnts, xlabel, log_scale=True):
	values = sorted(cnts)
	plt.figure()
	sns.histplot(values, stat='density', cumulative = True, 
				fill=False, element="step", log_scale=log_scale)
	plt.xlabel(xlabel)
	plt.ylabel("Frequency")
	plt.show()


if __name__ == "__main__":
	paritions = ["train_2020.json", "test_seen_2020.json", "test_unseen_2020.json"]
	parser = argparse.ArgumentParser()
	parser.add_argument("--partitions", type=list, default=[0])#, 1, 2])
	parser.add_argument("--force_rerun", type=bool, default=True)
	args = parser.parse_args()
	for p in args.partitions:
		summary_dict_file_name = paritions[p].split(".")[0] + "_summary.pkl"
		summary_dicts = None
		if os.path.exists(summary_dict_file_name) and not args.force_rerun:
			with open(summary_dict_file_name, 'rb') as f:
				summary_dicts = pickle.load(f)

		else:
			data = read_partition(paritions[p])
			summary_dicts = summarize_partition(paritions[p], data)
			with open(summary_dict_file_name, 'wb') as f:
				pickle.dump(summary_dicts, f)

		caption_lengths, speaker_counts, word_counts = summary_dicts
		count_hist(caption_lengths, "Captions Length in Words")
		frequency_hist(caption_lengths, "Captions Length in Words", log_scale=False)
		frequency_hist(speaker_counts.values(), "Captions Per Speaker")
		frequency_hist(word_counts.values(), "Word Counts")
	

