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

    all_texts = []
    for course in corpus['Course-list']:
        all_texts.append(course['CourseContent'])
        all_texts.extend(course['ILO-list-sv'])
    
    docs = nlp.bulk_process(all_texts)

    idx = 0
    for course in corpus['Course-list']:
        course['CourseContent'] = [token.lemma.lower() for sent in docs[idx].sentences for token in sent.words if token.pos != 'PUNCT' and token.lemma.lower() not in stop_words]
        idx += 1
        course['ILO-list-sv'] = [[token.lemma.lower() for token in docs[idx + j].sentences[0].words if token.pos != 'PUNCT' and token.lemma.lower() not in stop_words] for j in range(len(course['ILO-list-sv']))]
        idx += len(course['ILO-list-sv'])

    with open('data/SU.lemmatized.json', 'w') as f:
        json.dump(corpus, f, indent=2)

if __name__ == '__main__':
    main()