import json
import stanza
import nltk
from torch.backends import mps


def main():
    device = 'mps' if mps.is_available() else 'cpu'
    print(f'Using device: {device}')

    nlp = stanza.Pipeline(lang='sv', processors = 'tokenize,pos,lemma', device=device)
    nltk.download('stopwords')
    stop_words = set(nltk.corpus.stopwords.words('swedish'))

    with open('data/SU.filtered.json', 'r') as f:
        corpus = json.load(f)

    content_texts = [course['CourseContent'] for course in corpus['Course-list']]
    print('Processing course contents...')
    content_docs = nlp.bulk_process(content_texts)

    ilo_texts = [
        ilo
        for course in corpus['Course-list']
        for ilo in course['ILO-list-sv']
    ]
    print('Processing ILOs...')
    ilo_docs = nlp.bulk_process(ilo_texts)

    ilo_idx = 0
    for course, content_doc in zip(corpus['Course-list'], content_docs):

        course['CourseContent'] = [
            [
                token.lemma.lower()
                for token in sent.words
                if token.pos != 'PUNCT' and token.lemma and token.lemma.lower() not in stop_words
            ]
            for sent in content_doc.sentences
        ]

        n_ilos = len(course['ILO-list-sv'])
        course['ILO-list-sv'] = [
            [
                token.lemma.lower()
                for sent in ilo_docs[ilo_idx + j].sentences
                for token in sent.words
                if token.pos != 'PUNCT' and token.lemma and token.lemma.lower() not in stop_words
            ]
            for j in range(n_ilos)
        ]
        ilo_idx += n_ilos

    with open('data/SU.lemmatized.json', 'w') as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()