import json

from sklearn.feature_extraction.text import TfidfVectorizer

import torch.nn.functional as F
import torch

from sentence_transformers import SentenceTransformer

def main():
        
    def sbert():

        with open('data/SU.filtered.json', 'r') as f:
            corpus = json.load(f)
            
        model = SentenceTransformer('KBLab/sentence-bert-swedish-cased', device=device)

        course_info = []
        content_embeddings = []
        outcome_embeddings = []

        for course in corpus['Course-list']:
            course_info.append({'CourseCode': course['CourseCode'], 'Department': course['Department']})

            content_embeddings.append(
                model.encode(course['CourseContent'], convert_to_tensor=True, device=device)
            )
            outcomes = '. '.join(course['ILO-list-sv'])
            outcome_embeddings.append(model.encode(outcomes, convert_to_tensor=True, device=device))
            # outcome_embeddings.append(torch.mean(outcome_emb, dim=0))
        
        return course_info, content_embeddings, outcome_embeddings
        
    def tfidf():
        with open('data/SU.lemmatized.filtered.json', 'r') as f:
            corpus = json.load(f)

        def tokens2string(tokens):
            return ' '.join(tokens)

        all_texts = [tokens2string(course['CourseContent']) for course in corpus['Course-list']] + [tokens2string(outcome) for course in corpus['Course-list'] for outcome in course['ILO-list-sv']]

        print('Fitting TF-IDF vectorizer...')
        vectorizer = TfidfVectorizer()
        vectorizer.fit(all_texts)

        course_info = []
        content_embeddings = []
        outcome_embeddings = []

        for course in corpus['Course-list']:
            course_info.append({'CourseCode': course['CourseCode'], 'Department': course['Department']})
            content_embeddings.append(
                torch.tensor(vectorizer.transform([tokens2string(course['CourseContent'])]).toarray()[0], dtype=torch.float32)
            )
            outcomes = '. '.join([tokens2string(outcome) for outcome in course['ILO-list-sv']])
            outcome_embeddings.append(
                torch.tensor(vectorizer.transform([outcomes]).toarray()[0], dtype=torch.float32)
            )

            # outcome_vecs = []
            # for outcome in course['ILO-list-sv']:
            #     outcome_vecs.append(
            #         torch.tensor(vectorizer.transform([tokens2string(outcome)]).toarray()[0], dtype=torch.float32)
            #     )
            # outcome_embeddings.append(torch.mean(torch.stack(outcome_vecs), dim=0))
        
        return course_info, content_embeddings, outcome_embeddings

    def get_topk(similarity_matrix, k, course_info):

        topk = torch.topk(similarity_matrix, k=k, dim=1)

        results = {}

        for i, info in enumerate(course_info):
            course_code = info['CourseCode']
            results[course_code] = []

            for rank_idx in range(topk.indices.shape[1]):
                idx = topk.indices[i][rank_idx].item()
                score = topk.values[i][rank_idx].item()

                results[course_code].append({
                    "rank": rank_idx + 1,
                    "course_code": course_info[idx]['CourseCode'],
                    "dep": course_info[idx]['Department'],
                    "score": score
                })
        return topk, results
    
    def print_metrics(topk, course_info):
        top1_correct = 0
        top5_correct = 0
        mrr = 0

        for i, info in enumerate(course_info):
            course_code = info['CourseCode']
            indices = topk.indices[i]

            # Top-1
            if course_info[indices[0]]['CourseCode'] == course_code:
                top1_correct += 1

            # Top-5
            if course_code in [course_info[idx]['CourseCode'] for idx in indices]:
                top5_correct += 1

            rank = None
            for r, idx in enumerate(indices):
                if course_info[idx]['CourseCode'] == course_code:
                    rank = r + 1
                    break

            if rank is not None:
                mrr += 1 / rank

        top1_acc = top1_correct / len(course_info)
        top5_acc = top5_correct / len(course_info)
        mrr = mrr / len(course_info)

        print(f"Top-1 Accuracy: {top1_acc:.4f}")
        print(f"Top-5 Accuracy: {top5_acc:.4f}")
        print(f"MRR: {mrr:.4f}")

    # device = 'cpu'
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    method = 'hybrid'  # 'sbert', 'tfidf', or 'hybrid'
    reverse = True # If True, retrieves outcomes given content. If False, retrieves content given outcomes.

    if reverse:
        print("Retrieving outcomes given content...")
    else:
        print("Retrieving content given outcomes...")

    if method in ['sbert', 'tfidf']:
        if method == 'sbert':
            course_info, content_embeddings, outcome_embeddings = sbert()
        if method == 'tfidf':
            course_info, content_embeddings, outcome_embeddings = tfidf()
    
        content_embeddings = torch.stack(content_embeddings).to(device)
        outcome_embeddings = torch.stack(outcome_embeddings).to(device)

        # Normalize embeddings
        content_embeddings = F.normalize(content_embeddings, p=2, dim=1)
        outcome_embeddings = F.normalize(outcome_embeddings, p=2, dim=1)

        # Compute similarity matrix via matrix multiplication
        if reverse:
            similarity_matrix = torch.matmul(content_embeddings, outcome_embeddings.T)
        else:
            similarity_matrix = torch.matmul(outcome_embeddings, content_embeddings.T)

        topk, results = get_topk(similarity_matrix, k=5, course_info = course_info)
        print_metrics(topk, course_info)

    elif method == 'hybrid':
        course_info_sbert, content_embeddings_sbert, outcome_embeddings_sbert = sbert()
        course_info_tfidf, content_embeddings_tfidf, outcome_embeddings_tfidf = tfidf()

        assert course_info_sbert == course_info_tfidf

        course_info = course_info_sbert

        content_embeddings_sbert = F.normalize(torch.stack(content_embeddings_sbert), p=2, dim=1).to(device)
        outcome_embeddings_sbert = F.normalize(torch.stack(outcome_embeddings_sbert), p=2, dim=1).to(device)
        content_embeddings_tfidf = F.normalize(torch.stack(content_embeddings_tfidf), p=2, dim=1).to(device)
        outcome_embeddings_tfidf = F.normalize(torch.stack(outcome_embeddings_tfidf), p=2, dim=1).to(device)

        if reverse:
            sim_matrix_tfidf = torch.matmul(content_embeddings_tfidf, outcome_embeddings_tfidf.T)
            sim_matrix_sbert = torch.matmul(content_embeddings_sbert, outcome_embeddings_sbert.T)
        else:
            sim_matrix_tfidf = torch.matmul(outcome_embeddings_tfidf, content_embeddings_tfidf.T)
            sim_matrix_sbert = torch.matmul(outcome_embeddings_sbert, content_embeddings_sbert.T)

        sim_matrix_sbert = F.normalize(sim_matrix_sbert, p=2, dim=1)
        sim_matrix_tfidf = F.normalize(sim_matrix_tfidf, p=2, dim=1)

        for alpha in [0, 0.25, 0.5, 0.75, 1.0]:
            print(f"Hybridizing with alpha = {alpha}")
            print(f"Formula: sim = {alpha} * sbert + (1 - {alpha}) * tfidf")
            similarity_matrix = alpha * sim_matrix_sbert + (1 - alpha) * sim_matrix_tfidf
            topk, results = get_topk(similarity_matrix, k=5, course_info = course_info)
            print_metrics(topk, course_info)

    else:
        raise ValueError("Invalid method. Choose 'sbert', 'tfidf', or 'hybrid'.")

    if reverse:
        method = 'reverse.' + method
    with open(f'data/SU.{method}_retrieval.similarities.json', 'w') as f:
        json.dump(results, f, indent=2)

        
if __name__ == "__main__":
    main()