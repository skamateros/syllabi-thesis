import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    # device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # nlp = stanza.Pipeline(lang='sv', processors = 'tokenize,pos,lemma', device=device)
    # nltk.download('stopwords')
    # stop_words = set(nltk.corpus.stopwords.words('swedish'))

    # with open('data/SU.heuristics.json', 'r') as f:
    #     corpus = json.load(f)

    # print('Tokenizing and lemmatizing corpus...')

    # all_texts = []
    # for course in corpus['Course-list']:
    #     all_texts.append(course['CourseContent'])
    #     all_texts.extend(course['ILO-list-sv'])
    
    # docs = nlp.bulk_process(all_texts)

    # idx = 0
    # for course in corpus['Course-list']:
    #     course['CourseContent'] = [token.lemma.lower() for sent in docs[idx].sentences for token in sent.words if token.pos != 'PUNCT' and token.lemma.lower() not in stop_words]
    #     idx += 1
    #     course['ILO-list-sv'] = [[token.lemma.lower() for token in docs[idx + j].sentences[0].words if token.pos != 'PUNCT' and token.lemma.lower() not in stop_words] for j in range(len(course['ILO-list-sv']))]
    #     idx += len(course['ILO-list-sv'])

    with open('data/SU.lemmatized.json', 'r') as f:
        corpus = json.load(f)

    all_texts = []
    for course in corpus['Course-list']:
        all_texts.append(course['CourseContent'])
        all_texts.extend(course['ILO-list-sv'])

    def tokens2string(tokens):
        return ' '.join(tokens)

    all_texts = [tokens2string(course['CourseContent']) for course in corpus['Course-list']] + [tokens2string(outcome) for course in corpus['Course-list'] for outcome in course['ILO-list-sv']]

    print('Fitting TF-IDF vectorizer...')
    vectorizer = TfidfVectorizer()
    vectorizer.fit(all_texts)

    print('Calculating similarities...')
    output = {}
    for course in corpus['Course-list']:
        course_vec = vectorizer.transform([tokens2string(course['CourseContent'])])

        similarities = []
        for outcome in course['ILO-list-sv']:
            outcome_vec = vectorizer.transform([tokens2string(outcome)])
            similarity = cosine_similarity(course_vec, outcome_vec)[0][0]
            similarities.append(similarity)
        
        output[course['CourseCode']] = similarities

    print('Saving similarities to JSON file...')
    with open('data/SU.tfidf.similarities.json', 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == '__main__':
    main()