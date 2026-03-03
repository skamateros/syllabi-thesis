import gensim
import json
import nltk
import stanza

import torch
from torch.nn.functional import cosine_similarity

def main():
    nltk.download('stopwords')
    stop_words = set(nltk.corpus.stopwords.words('swedish'))

    with open('data/SU.heuristics.json', 'r') as f:
        corpus = json.load(f)

    def tokenize(text):
        tokens = gensim.utils.simple_preprocess(text)
        tokens = [token for token in tokens if token not in stop_words]
        return tokens

    for course in corpus['Course-list']:
        course['CourseContent'] = tokenize(course['CourseContent'])
        course['ILO-list-sv'] = [tokenize(outcome) for outcome in course['ILO-list-sv']]

    texts = [course['CourseContent'] for course in corpus['Course-list']] + [outcome for course in corpus['Course-list'] for outcome in course['ILO-list-sv']]
    id2word = gensim.corpora.Dictionary(texts)
    model_corpus = [id2word.doc2bow(text) for text in texts]


    lda_model = gensim.models.ldamodel.LdaModel(corpus=model_corpus,
                                            id2word=id2word,
                                            num_topics=20, 
                                            random_state=19,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=False)

    # topics = lda_model.print_topics(num_words=10)
    # for topic in topics:
    #     print(topic)
    
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

    with open('data/SU.lda.similarities.json', 'w') as f:
        json.dump(output, f, indent=2, cls=TensorEncoder)

    # print(corpus['Course-list'][0])

if __name__ == '__main__':
    main()