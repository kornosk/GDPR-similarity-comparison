#!/usr/bin/env python
# coding: utf-8

'''
@title: Get Top K of Each Article
@authors: Kornraphop Kawintiranon (Ken)
@institution: Georgetown University
@project: GDPR-similarity-comparison
@date: April 2021
@description: Get top K related articles of each EU GDPR article ranked by similarity scores, write to file.
'''

import pandas as pd
import numpy as np
import csv


def main():
	k = 1

	for method in ["tfidf", "glove", "bert", "siamese"]:
		input_filepath = "data/simi_article_{}.csv".format(method)
		output_filepath = "top_k_each_article/simi_article_{}_top{}.csv".format(method, k)

		df = pd.read_csv(input_filepath, sep="~")
		print(f"Columns: {df.columns}")
		print(f"There are {df.shape} pairs.")

		# Get top K of each article
		df_top_k = df.groupby('gdpr article').head(k).reset_index(drop=True)

		# Write to file
		df_top_k.to_csv(output_filepath, index=False)


if __name__ == '__main__':
	main()