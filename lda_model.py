import gensim
import json
import nltk
import stanza

import torch
from torch.nn.functional import cosine_similarity

def main():
    nlp = stanza.Pipeline(lang='sv', processors = 'tokenize,pos,lemma')

    nltk.download('stopwords')
    stop_words = set(nltk.corpus.stopwords.words('swedish'))

    with open('data/SU.heuristics.json', 'r') as f:
        corpus = json.load(f)

    print('Tokenizing and lemmatizing corpus...')

    all_texts = []
    for course in corpus['Course-list']:
        # course['CourseContent'] = tokenize(course['CourseContent'])
        # course['ILO-list-sv'] = [tokenize(outcome) for outcome in course['ILO-list-sv']]
        all_texts.append(course['CourseContent'])
        all_texts.extend(course['ILO-list-sv'])
    
    docs = nlp.bulk_process(all_texts)

    idx = 0
    for course in corpus['Course-list']:
        course['CourseContent'] = [token.lemma.lower() for sent in docs[idx].sentences for token in sent.words if token.pos != 'PUNCT' and token.lemma.lower() not in stop_words]
        idx += 1
        course['ILO-list-sv'] = [[token.lemma.lower() for token in docs[idx + j].sentences[0].words if token.pos != 'PUNCT' and token.lemma.lower() not in stop_words] for j in range(len(course['ILO-list-sv']))]
        idx += len(course['ILO-list-sv'])

    texts = [course['CourseContent'] for course in corpus['Course-list']] + [outcome for course in corpus['Course-list'] for outcome in course['ILO-list-sv']]

    id2word = gensim.corpora.Dictionary(texts)
    model_corpus = [id2word.doc2bow(text) for text in texts]

    print('Training LDA model...')
    lda_model = gensim.models.ldamodel.LdaModel(corpus=model_corpus,
                                            id2word=id2word,
                                            num_topics=20, 
                                            random_state=19,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=False)

    
    print('Calculating similarities...')
    output = {}
    for course in corpus['Course-list']:
        course_bow = id2word.doc2bow(course['CourseContent'])
        course_topics = lda_model.get_document_topics(course_bow)

        for i in range(lda_model.num_topics):
            if i not in [topic_num for topic_num, _ in course_topics]:
                course_topics.append((i, 0.0))

        course_topics.sort(key=lambda x: x[0])  # Sort by topic number
        course_topic_distribution = [topic[1] for topic in course_topics]

        # print(f"Course: {course['CourseCode']}")
        # print(f"Course content topic distribution: {course_topics}")

        similarities = []
        for outcome in course['ILO-list-sv']:
            outcome_bow = id2word.doc2bow(outcome)
            outcome_topics = lda_model.get_document_topics(outcome_bow)

            for i in range(lda_model.num_topics):
                if i not in [topic_num for topic_num, _ in outcome_topics]:
                    outcome_topics.append((i, 0.0))

            outcome_topics.sort(key=lambda x: x[0])  # Sort by topic number
            outcome_topic_distribution = [topic[1] for topic in outcome_topics]

            assert len(course_topic_distribution) == len(outcome_topic_distribution), "Topic distributions must have the same length"

            # print(f"Outcome: {outcome}")
            # print(f"Outcome topic distribution: {outcome_topics}")
            similarities.append(cosine_similarity(torch.tensor(course_topic_distribution), torch.tensor(outcome_topic_distribution), dim=0).item())
        output[course['CourseCode']] = similarities

    # Custom JSON Encoder to handle pytorch tensors
    class TensorEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            return super().default(obj)

    print('Saving similarities to JSON...')
    with open('data/SU.lda.similarities.json', 'w') as f:
        json.dump(output, f, indent=2, cls=TensorEncoder)

if __name__ == '__main__':
    main()