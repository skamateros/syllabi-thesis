import json

from sklearn.feature_extraction.text import TfidfVectorizer

import torch.nn.functional as F
import torch

from sentence_transformers import SentenceTransformer

def main():
        
    def sbert():
        # device = 'cpu'
        device = torch.device('mps' if torch.mps.is_available() else 'cpu')
        print(f'Using device: {device}')

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

            outcome_emb = model.encode(course['ILO-list-sv'], convert_to_tensor=True, device=device)
            outcome_embeddings.append(torch.mean(outcome_emb, dim=0))
        
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
            outcome_vecs = []
            for outcome in course['ILO-list-sv']:
                outcome_vecs.append(
                    torch.tensor(vectorizer.transform([tokens2string(outcome)]).toarray()[0], dtype=torch.float32)
                )
            outcome_embeddings.append(torch.mean(torch.stack(outcome_vecs), dim=0))
        
        return course_info, content_embeddings, outcome_embeddings

    method = 'sbert'  # 'sbert', 'tfidf', or 'hybrid'

    if method in ['sbert', 'tfidf']:
        if method == 'sbert':
            course_info, content_embeddings, outcome_embeddings = sbert()
        if method == 'tfidf':
            course_info, content_embeddings, outcome_embeddings = tfidf()
    
        content_embeddings = torch.stack(content_embeddings)
        outcome_embeddings = torch.stack(outcome_embeddings)

        # Normalize embeddings
        content_embeddings = F.normalize(content_embeddings, p=2, dim=1)
        outcome_embeddings = F.normalize(outcome_embeddings, p=2, dim=1)

        # Compute similarity matrix via matrix multiplication
        similarity_matrix = torch.matmul(outcome_embeddings, content_embeddings.T)

    elif method == 'hybrid':
        course_info, content_embeddings_sbert, outcome_embeddings_sbert = sbert()
        course_info, content_embeddings_tfidf, outcome_embeddings_tfidf = tfidf()

        content_embeddings_sbert = F.normalize(torch.stack(content_embeddings_sbert), p=2, dim=1)
        outcome_embeddings_sbert = F.normalize(torch.stack(outcome_embeddings_sbert), p=2, dim=1)
        content_embeddings_tfidf = F.normalize(torch.stack(content_embeddings_tfidf), p=2, dim=1)
        outcome_embeddings_tfidf = F.normalize(torch.stack(outcome_embeddings_tfidf), p=2, dim=1)

        sim_matrix_tfidf = torch.matmul(outcome_embeddings_tfidf, content_embeddings_tfidf.T)
        sim_matrix_sbert = torch.matmul(outcome_embeddings_sbert, content_embeddings_sbert.T)

        alpha = 0.5  # 50/50 weighting
        similarity_matrix = alpha * sim_matrix_sbert + (1 - alpha) * sim_matrix_tfidf

    else:
        raise ValueError("Invalid method. Choose 'sbert', 'tfidf', or 'hybrid'.")


    topk = torch.topk(similarity_matrix, k=5, dim=1)

    results = {}

    for i, info in enumerate(course_info):
        course_code = info['CourseCode']
        results[course_code] = []

        for k in range(topk.indices.shape[1]):
            idx = topk.indices[i][k].item()
            score = topk.values[i][k].item()

            results[course_code].append({
                "rank": k + 1,
                "course_code": course_info[idx]['CourseCode'],
                "dep": course_info[idx]['Department'],
                "score": score
            })

    with open(f'data/SU.{method}_retrieval.similarities.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Calculate Top-1 and Top-5 accuracy
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
    mrr /= len(course_info)

    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")
    print(f"MRR: {mrr:.4f}")

        
if __name__ == "__main__":
    main()