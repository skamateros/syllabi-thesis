import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tokens2string(tokens):
    return ' '.join(tokens)

def main():
    with open('data/SU.lemmatized.json', 'r') as f:
        corpus = json.load(f)

    match_per = 'outcome'  # 'content' or 'outcome'

    # Fit on all content chunks and all ILOs
    all_texts = (
        [tokens2string(chunk) for course in corpus['Course-list'] for chunk in course['CourseContent']] +
        [tokens2string(outcome) for course in corpus['Course-list'] for outcome in course['ILO-list-sv']]
    )

    print('Fitting TF-IDF vectorizer...')
    vectorizer = TfidfVectorizer()
    vectorizer.fit(all_texts)

    print('Calculating similarities...')
    output = {}
    for course in corpus['Course-list']:
        course_code = course['CourseCode']

        content_chunks = [tokens2string(chunk) for chunk in course['CourseContent']]
        course_outcomes = [tokens2string(outcome) for outcome in course['ILO-list-sv']]

        if not content_chunks or not course_outcomes:
            if match_per == 'content':
                output[course_code] = {
                    'outcomes': course_outcomes,
                    'chunks': []
                }
            elif match_per == 'outcome':
                output[course_code] = {
                    'content_chunks': content_chunks,
                    'outcomes': []
                }
            continue

        content_chunk_vecs = vectorizer.transform(content_chunks)
        outcome_vecs = vectorizer.transform(course_outcomes)

        if match_per == 'content':
            output[course_code] = {
                'outcomes': course_outcomes,
                'chunks': []
            }

            for chunk_idx, chunk in enumerate(content_chunks):
                chunk_vec = content_chunk_vecs[chunk_idx]

                sim_list = [cosine_similarity(chunk_vec, outcome_vecs[i])[0][0] for i in range(outcome_vecs.shape[0])]

                best_idx = sim_list.index(max(sim_list))
                best_sim = sim_list[best_idx]

                output[course_code]['chunks'].append({
                    'chunk': chunk,
                    'max_similarity': best_sim,
                    'outcome_index': best_idx,
                    'matched_outcome': course_outcomes[best_idx]
                })

        elif match_per == 'outcome':
            output[course_code] = {
                'content_chunks': content_chunks,
                'outcomes': []
            }

            for outcome_idx, outcome in enumerate(course_outcomes):
                outcome_vec = outcome_vecs[outcome_idx]

                sim_list = [
                    cosine_similarity(outcome_vec, content_chunk_vecs[i])[0][0]
                    for i in range(content_chunk_vecs.shape[0])
                ]

                best_idx = sim_list.index(max(sim_list))
                best_sim = sim_list[best_idx]

                output[course_code]['outcomes'].append({
                    'outcome': outcome,
                    'max_similarity': best_sim,
                    'chunk_index': best_idx,
                    'matched_chunk': content_chunks[best_idx]
                })

        else:
            raise ValueError("Invalid match_per value. Use 'content' or 'outcome'.")

    print('Saving similarities to JSON file...')
    with open(f'data/{match_per}.tfidf.similarities.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()