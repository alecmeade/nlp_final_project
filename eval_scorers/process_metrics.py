import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
import collections
import os
import pickle
import seaborn as sns
import pandas as pd

from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def read_metric_files(metrics_base_dir, metric_file_names):
	metrics_dict = {}
	for f_path in metric_file_names:
		with open(os.path.join(metrics_base_dir, f_path), 'r') as f:
			metrics_dict[f_path] = pd.read_json(f, orient='index')

	return metrics_dict



def top_k_precision(y, logits, labels, k):
	sorted_idx = np.argsort(logits, axis = 1)
	y_pred_top_k = np.zeros(len(y))
	for i in range(len(y)):
		if y[i] in sorted_idx[i, -k:]:
			y_pred_top_k[i] = y[i]
		else:
			y_pred_top_k[i] = sorted_idx[i, 0]

	return precision_score(y, y_pred_top_k, labels=labels, average='micro')


def top_k_recall(y, logits, labels, k):
	sorted_idx = np.argsort(logits, axis = 1)
	y_pred_top_k = np.zeros(len(y))
	for i in range(len(y)):
		if y[i] in sorted_idx[i, -k:]:
			y_pred_top_k[i] = y[i]
		else:
			y_pred_top_k[i] = sorted_idx[i, 0]


	return recall_score(y, y_pred_top_k, labels=labels, average='micro')


def summarize_metrics(name, metrics_df, top_n = [1, 2, 3, 4, 5]):
	summary = {'name': name}

	logits = np.array([l for l in metrics_df.logits])
	y = np.array([l for l in metrics_df.y])
	y_pred = np.array([l for l in metrics_df.y_pred])
	labels = list(range(0, 205))
	for k in top_n:
		summary['top_%d_accuracy' % k] = top_k_accuracy_score(y, logits, labels = labels, k=k)
		
		precision = top_k_precision(y, logits, labels, k)
		summary['top_%d_precions' % k] = precision 

		recall = top_k_recall(y, logits, labels, k)
		summary['top_%d_recall' % k] = recall 
		
		summary['top_%d_f1' % k] = 2 * (precision * recall) / (precision + recall)


	summary['acc'] = accuracy_score(metrics_df.y, metrics_df.y_pred)
	summary['precision'] = precision_score(metrics_df.y, metrics_df.y_pred, average='micro')
	summary['recall'] = recall_score(metrics_df.y, metrics_df.y_pred, average='micro')
	summary['f1'] = f1_score(metrics_df.y, metrics_df.y_pred, average='micro')


	for m in ['sisa', 'misa', 'sima']:
		summary["%s_mean" % m] = metrics_df[m].mean()
		summary["%s_std" % m] = metrics_df[m].std()
	

	return summary

def print_summary(summary):
	for k, v in summary.items():
		print(k, v)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--metrics_base_dir", type=str, default='results')
	parser.add_argument("--metric_file_names", type=list, default=['results.json'])
	
	args = parser.parse_args()
	metrics_dict = read_metric_files(args.metrics_base_dir, args.metric_file_names)

	for name, metrics_df in metrics_dict.items():
		print_summary(summarize_metrics(name, metrics_df))
		print("*******************************\n")









