import argparse
import copy
import json
import numpy as np
from apricot import FeatureBasedSelection
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline


def read_partition(path):
	data = None
	with open(path, 'r') as f:
		data = json.loads(f.read())

	return data

if __name__ == "__main__":
	paritions = ["train_2020.json", "test_seen_2020.json", "test_unseen_2020.json"]
	parser = argparse.ArgumentParser()
	parser.add_argument("--partition", type=int, default=0)
	parser.add_argument("--coreset_frac", type=float, default=0.1)
	parser.add_argument("--display_tsne", type=bool, default=True)
	parser.add_argument("--tsne_frac", type=float, default=0.03)
	args = parser.parse_args()
	partition = paritions[args.partition] 
	data = read_partition(partition)
	corpus = []
	Y = []
	vocabulary = {}
	for d in data['data']:
		corpus.append(d['asr_text'])
		for w in d['asr_text']:
			vocabulary[w] = True
		Y.append(d['uttid'])

	Y = np.array(Y)
	vocabulary = vocabulary.keys()

	print("TFIDF Conversion...")
	pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),
		('tfid', TfidfTransformer())]).fit(corpus)
	X = pipe.transform(corpus)
	print("TFIDF Converted.")


	print("Identifying Core Set...")
	total_samples = X.shape[0]
	n_samples = int(total_samples * args.coreset_frac)
	fsb = FeatureBasedSelection(n_samples)
	core_X, core_Y = fsb.fit_transform(X, Y)
	core_Y = set(core_Y)
	
	print("Core Set Generated.")
	print("Total Samples:", total_samples)
	print("Core Set Samples:", n_samples)

	if args.display_tsne:
		print("Displaying Data in TSNE...")
		tsne_samples = int(total_samples * args.tsne_frac)
		inds = np.random.choice(Y.shape[0], tsne_samples, replace = False)
		tsne = TSNE(n_components=2, perplexity=20, random_state=0)
		X_2d = tsne.fit_transform(X[inds, :])


		plt.figure(figsize=(10, 10))
		
		colors = []
		for y in Y[inds]:
			if y in core_Y:
				colors.append('r')
			else:
				colors.append('b')


		plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.7)
		plt.legend()
		plt.show()


	new_partition = copy.deepcopy(data)
	new_partition['data'] = []

	for d in data['data']:
		if d['uttid'] in core_Y:
			new_partition['data'].append(d)


	new_partition_name = partition.split('.')[0] + "_core.json"
	with open(new_partition_name, 'w') as f:
		json.dump(new_partition, f)

