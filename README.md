# Computational Analysis of Alignment in Course Syllabi
Exploring intra-document alignment using lexical and embedding-based methods

## Paper

The full paper will soon be published on DiVA.

## Abstract

Manually reviewing large collections of university course syllabi is time-consuming, and standardising evaluation criteria is challenging.
This creates an opportunity for NLP methods to aid syllabus review processes, particularly in relation to constructive alignment and student-centred learning. This study investigates the extent to which lexical and embedding-based methods of semantic representation can be used to assess alignment between course content and intended learning outcomes in Swedish university syllabi. The analysis is based on 3,612 syllabi from Stockholm University over the period of 2007-2024. A lexical (TF-IDF) and an embedding-based (KB-SBERT) text representation approach are compared for computing similarity scores. The methods are evaluated using correlation to human judgements, as well as a retrieval task. Results show that the two approaches capture different aspects of alignment: KB--SBERT correlates more strongly with human judgements, while TF-IDF, benefitting from the domain-specific vocabulary present in syllabi, achieves higher retrieval performance. Systematic variation in similarity scores was found across university departments, suggesting that writing conventions, as well as structural and administrative characteristics of syllabi influence how alignment is measured computationally. Finally, a qualitative error analysis highlighted certain model limitations and pipeline sensitivities related to pre-processing and segmentation strategies. Overall, the findings indicate that NLP methods can support quality assurance processes at scale, but require critical human interpretation and awareness of methodological assumptions and limitations.

## Scripts

- filter.py - Discards instances of courses based on the criteria described in §3.1.1
- lemmatize.py - Lemmatisation and stop word removal for the TF-IDF implementation baseline.
- tfidf.py - TF-IDF implementation.
- sbert.py - KB-SBERT implementation.
- correlation.py - Correlation calculation script that supports spearman and pearson correlation.
- outliers.py - Extracts outliers for each method and matching direction.
