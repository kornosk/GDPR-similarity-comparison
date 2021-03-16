""" For processing GDPR documents into a csv file
"""

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
	lines = [p.text.strip() for p in paragraphs]

	return lines


def main():
	filepath = "GDPR-EN-Europe-converted.docx"
	target_filepath = "GDPR-EN-Europe-converted.csv"
	
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
				# print("yeah")
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
			
	# print("Dataframe Size:", df.shape)
	# print(df.head(30))

	# Write to file
	df.to_csv(target_filepath, index=False)

if __name__ == '__main__':
	main()