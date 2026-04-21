import json
import numpy as np

def get_threshold(sims):
    sims_array = np.array(sims)
    mean = np.mean(sims_array)
    std_dev = np.std(sims_array)
    threshold = mean - 2 * std_dev
    return threshold

for i in ['outcome', 'content']:
    for j in ['tfidf', 'sbert']:
        with open(f'../data/{i}.{j}.similarities.json', 'r') as f:
            data = json.load(f)

        threshold = get_threshold([
            chunk['max_similarity'] 
            for course in data.values() 
            for chunk in (course['chunks'] if i == 'content' else course['outcomes'])
            ])
        
        if threshold < 0:
            threshold = 0.0

        print(f"Threshold for {i} with {j}: {threshold}")
        output = []
        for key, course in data.items():
            
            if i == 'content':
                for chunk in course['chunks']:
                    if chunk['max_similarity'] <= threshold:
                        output.append({
                            "course_code": key,
                            "chunk": chunk['chunk'],
                            "similarity": chunk['max_similarity'],
                            "matched_outcome": chunk['matched_outcome']
                        })
            elif i == 'outcome':
                for outcome in course['outcomes']:
                    if outcome['max_similarity'] <= threshold:
                        output.append({
                            "course_code": key,
                            "outcome": outcome['outcome'],
                            "similarity": outcome['max_similarity'],
                            "matched_chunk": outcome['matched_chunk']
                        })
            else:
                raise ValueError("Invalid i value. Use 'content' or 'outcome'.")
            
        with open(f'../data/outliers.{i}.{j}.json', 'w') as f:
            json.dump(output, f, indent=2)