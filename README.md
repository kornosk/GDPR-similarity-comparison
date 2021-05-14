# GDPR Similarity Comparison
We extract information from GDPR-like documents from different countries written in natuaral language and construct well-strucured data.

The structured data are 4 columns including `chapter, section, article and recital`. This could benefit any future work that would like to explore GDPR-like using computational methods. 🚀

This project is inspired by [COSC-824 Data Protection by Design](https://courses.benujcich.georgetown.domains/cosc824/sp2021/), Department of Computer Science at Georgetown University.

## Data
We convert from PDF to Docx to CSV with well-structured style. Now, our data include GDPR-like documents from:
* European 🇪🇺
* Brazil 🇧🇷
* Indian 🇮🇳
* What next? 😉

Simply load the data into a dataframe in `Python` as following code.

```python
import pandas as pd

file_path = "data/LGPD-ES-Brazil-converted.csv"
df = pd.read_csv(file_path) # columns: ["chapter", "section", "article", "recital"]
```

## Materials
* [Presentation slide](https://docs.google.com/presentation/d/12DSfG_3TE1FYhimZhktm2EmdRjRjoBQ0K9QxeT-fhyQ/edit?usp=sharing)
* Preprint - Coming soon 🔥

## Project Member
* Kornraphop Kawintiranon - [Github](https://github.com/kornosk)
* Yaguang Liu - [Github](https://github.com/steveguang)
* Prof. Benjamin E. Ujcich (Instructor) - [Personal](https://personal.benujcich.georgetown.domains/)

## Citation
If you feel our paper and resources are useful and encouraging, please consider citing our work! 🙏
```bibtex
Coming soon
```

## References
- PDF to Docx: [https://smallpdf.com/pdf-to-word](https://smallpdf.com/pdf-to-word)
