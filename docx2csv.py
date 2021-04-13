#!/usr/bin/env python
# coding: utf-8

'''
@title: doc2vec
@authors: Kornraphop Kawintiranon (Ken)
@institution: Georgetown University
@project: GDPR-similarity-comparison
@date: April 2021
@description: For processing GDPR documents into a csv file
'''

import re
import numpy as np
import pandas as pd
from docx import Document


def read_docx(filepath):
	""" Read docx file and return all lines of texts
	"""

	document = Document(filepath)
	
	sections = document.sections
	print("Section Size:", len(sections))

	paragraphs = document.paragraphs
	print("Paragraphs Size:", len(paragraphs))

	# stripe spaces
	p_lines = [p.text.strip() for p in paragraphs]

	return p_lines


def extract_gdpr_eu(filepath: str):
	"""	Extract natural language on GDPR European version
		to smaller components including "chapter", "section", "article" and "recital"
	"""

	# read docx to lines
	lines = read_docx(filepath)

	for idx, line in enumerate(lines[:20]):
		print(f"Line {idx}: {line}")

	chapters = ["undefined"]
	articles = ["undefined"]
	sections = ["undefined"]

	df = pd.DataFrame(columns=["chapter", "section", "article", "recital"])

	processing_chapter = False
	processing_section = False
	processing_article = False
	# processing_recital = False

	tmp_str = ""
	for idx in range(len(lines)):
		line = lines[idx].strip()

		if line.startswith("CHAPTER "):
			# chapters.append(line.strip())
			processing_chapter = True
			processing_section = False
			processing_article = False
			tmp_str = line
			
			# start new articles and sections
			articles = ["undefined"]
			sections = ["undefined"]

		elif line.startswith("Section "):
			processing_chapter = False
			processing_section = True
			processing_article = False
			tmp_str = line
		
		elif line.startswith("Article "):
			processing_chapter = False
			processing_section = False
			processing_article = True

		elif len(line):
			if processing_chapter:
				tmp_str = tmp_str.strip() + " " + line.strip()
			elif processing_section:
				tmp_str = tmp_str.strip() + " " + line.strip()
				# sections.append(line.strip())
			elif processing_article:
				processing_chapter = False
				processing_section = False
				processing_article = False
				articles.append(line.strip())
			else:
				row = [chapters[-1], sections[-1], articles[-1], line.strip()]
				df.loc[len(df.index)] = row
		else: # len(line) <= 0
			if processing_chapter:
				processing_chapter = False
				processing_section = False
				processing_article = False
				chapters.append(tmp_str.strip())
				# print("Chapter:", tmp_str.strip())
				tmp_str = ""
			elif processing_section:
				processing_chapter = False
				processing_section = False
				processing_article = False
				sections.append(tmp_str.strip())
				# print("Chapter:", tmp_str.strip())
				tmp_str = ""
			# elif processing_article:
			# 	processing_article = False
			# 	articles.append(tmp_str.strip())
			# 	tmp_str = ""
	
	return df


def extract_gdpr_indian(filepath: str, start_line=185):
	"""	Extract natural language on GDPR Indian version
		to smaller components including "chapter", "clause", and "recital"
	"""

	# read docx to lines
	lines = read_docx(filepath)

	for idx, line in enumerate(lines[185:210]):
		print(f"Line {idx}: {line}")

	chapters = ["undefined"]
	clauses = ["undefined"]

	df = pd.DataFrame(columns=["chapter", "clause", "recital"])

	processing_chapter = False
	processing_clause = False

	tmp_str = ""
	for idx in range(start_line, len(lines)):
		line = lines[idx].strip()
		
		# Some lines include line number with indent
		pattern = re.compile(r"^\d+\t+.+")
		if pattern.match(line) is not None:
			line = re.sub(r"\t+", "\t", line)
			line = line[line.index("\t") + 1:]

		# Capture the chapter
		if line.startswith("CHAPTER "):

			# if chapter name is in the same line
			if len(line) > 12:
				processing_chapter = False
				processing_clause = False
				chapters.append(line.strip())
				tmp_str = ""

			# if chapter name is in the next line
			else:
				processing_chapter = True
				processing_clause = False
				tmp_str = line
			
			# start new clause
			clause = ["undefined"]

		elif len(line) > 5:

			# some special exception
			# e.g. 53 of 2005.
			pattern = re.compile(r'^\d+ of \d+\.( \d+)?$')
			if pattern.match(line) is not None:
				continue

			# add chapter name to the chapter list
			if processing_chapter:
				tmp_str = tmp_str.strip() + " " + line.strip()
				processing_chapter = False
				processing_clause = False
				chapters.append(tmp_str.strip())
				tmp_str = ""

			# add recital to dataframe
			else:
				# some exceptions
				pattern = re.compile(r"^\d+ .+$")
				if pattern.match(line) is not None:
					line = line[line.index(" ") + 1:]

				row = [chapters[-1], clauses[-1], line.strip()]
				df.loc[len(df.index)] = row
		else: # len(line) <= 0
			if processing_chapter:
				processing_chapter = False
				processing_clause = False
				chapters.append(tmp_str.strip())
				tmp_str = ""
			elif processing_clause:
				processing_chapter = False
				processing_clause = False
				clauses.append(tmp_str.strip())
				tmp_str = ""

	return df


def main():
	# filepath = "GDPR-EN-Europe-converted.docx"
	filepath = "GDPR-EN-Indian-converted.docx"
	# target_filepath = "GDPR-EN-Europe-converted.csv"
	target_filepath = "GDPR-EN-Indian-converted.csv"

	# df = extract_gdpr_eu(filepath)
	df = extract_gdpr_indian(filepath)
			
	# print("Dataframe Size:", df.shape)
	# print(df.head(30))

	# Write to file
	df.to_csv(target_filepath, index=False)


if __name__ == '__main__':
	main()