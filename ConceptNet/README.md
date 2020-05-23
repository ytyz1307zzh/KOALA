# ConceptNet Retrieval

The original ConceptNet knowledge base is stored in csv format, which is provided [here](https://github.com/commonsense/conceptnet5/wiki/Downloads).

1. Obtain the English subgraph: the original ConceptNet is multi-lingual, so we only keep the English triples. This means both the subject and the object contain only English letters. 

2. Prune the knowledge graph: restrict the relation set to those mentioned in `relation_direction.txt`, remove stop words in the original triple, remove invalid and meaningless triples, and adjust the weight of specific types of triples (`relatedto` and `antonym` triples receives an additional -0.3, `atlocation` triples receives an additional +0.5) in order to better fit our retrieval target.

3. Rough retrieval: retrieve those triples whose subject or object matches with the given entity. For some relations, the entity must be the subject but for others, either subject or object is ok (see `relation_direction.txt` for detail). Stemming is used to perform soft matching. We remove stop words before matching. If the entity is a single word, then we directly use string matching. If the entity is a phrase, we keep those entity-concept pairs with Jaccard similarity >= 0.5. The concept should be a noun (if POS is specified).

4. Precise retrieval: we use the paragraph to perform precise retrieval. After removing stop words, we use the paragraph to match the triples extracted from the last step. 

   (1) Exact match: If the neighbor concept (to the entity) is contained in the paragraph (Jaccard similarity >= 0.5 for phrasal concepts), then we keep this triple. The retrieved triples are sorted by weight and top 10 are selected. 

   (2) Fuzzy match: If the retrieved exact match triples are less than 10, then we apply fuzzy match as a complement. Knowledge triples and text paragraphs are encoded by BERT. The maximum cosine similarity of each content word in the paragraph to the neighboring concept in the triple is used as the semantic relevance between the paragraph and the triple. Triples are sorted by semantic relevance and the top ones are selected to make up a total number of 10 triples.

## Scripts

`extract_english.py`: extract the English subgraph from ConceptNet.

`prune_graph.py`: remove invalid triples and adjust the weights of some specific relations.

`rough_retrieval.py`: extract entity-centric subgraph (all triples related to a given entity) from ConceptNet

`exact_match.py`: select knowledge triples based on exact match

`fuzzy_match_word.py`: sort knowledge triples based on fuzzy match scores (BERT word embeddings)

`fuzzy_match_sent.py`: sort knowledge triples based on fuzzy match scores (BERT sentence embeddings). Note that this script is not used in our final model.

`select_state_verb.py`: select high-frequency co-appearance verbs to state "create", "move" and "destroy" from the training set.

`combine_exact_fuzzy.py`: combine the exact-match triples and fuzzy-match triples to make up at most 10 triples for each data instance.

## Auxiliary Files

``relation_direction.txt``: this file indicates the valid relations as well as their valid directions (for the given entity) considered in our work.

`triple2sent.txt`: the rulebase of transforming knowledge triples to natural language sentences.

`Stemmer.py`: The Porter Stemmer used in word stemming.



