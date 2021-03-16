"""
For testing the library - python-docx
"""
	
from docx import Document


def main():
	document = Document('GDPR-EN-Europe-converted.docx')

	sections = document.sections
	print("Section Size:", len(sections))

	paragraphs = document.paragraphs
	print("Paragraphs Size:", len(paragraphs))

	for idx, p in enumerate(paragraphs[:20]):
		print(f"Line {idx}: {p.text}")


if __name__ == '__main__':
	main()