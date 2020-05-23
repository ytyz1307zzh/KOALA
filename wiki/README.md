# Wiki Retrieval

Retrieve Wiki paragraphs from wikipedia dump using TF-IDF.

1. Acquire the wiki dump and the retriever model according to the [DrQA](https://github.com/facebookresearch/DrQA) repo.

2. Convert the doc-level wiki database to paragraph-level using the `ConvertWikiPage2Paragraph` class in `retrieve_para.py`
3. Use `prepare_input.py` to format the query inputs of wiki retrieval.
4. Use `retrieve_para.py` to retrieve top 50 wiki relevant paragraphs to each instance.

P.S. DrQA modules are included in this directory. The retrieved wiki paragraphs are stored in `wiki_para_50.json`.
