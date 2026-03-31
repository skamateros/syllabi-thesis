import json

for i in ['outcome', 'content']:

    with open(f'data/SU.{i}.sbert.similarities.json', 'r') as f:
        sbert_sims = json.load(f)

    output = []
    for key, course in sbert_sims.items():
        if i == 'content':
            for chunk in course['chunks']:
                if chunk['max_similarity'] < 0.35:
                    output.append({
                        "course_code": key,
                        "chunk": chunk['chunk'],
                        "similarity": chunk['max_similarity'],
                        "matched_outcome": chunk['matched_outcome']
                    })
        elif i == 'outcome':
            for outcome in course['outcomes']:
                if outcome['max_similarity'] < 0.35:
                    output.append({
                        "course_code": key,
                        "outcome": outcome['outcome'],
                        "similarity": outcome['max_similarity'],
                        "matched_chunk": outcome['matched_chunk']
                    })
        else:
            raise ValueError("Invalid i value. Use 'content' or 'outcome'.")
        
    with open(f'data/outliers.{i}.json', 'w') as f:
        json.dump(output, f, indent=2)