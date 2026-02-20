import numpy as np
import json

from sentence_transformers import SentenceTransformer

def main():
    with open('data/SU.heuristics.json', 'r') as f:
        corpus = json.load(f)
    
    model = SentenceTransformer('KBLab/sentence-bert-swedish-cased')

    # print(corpus['Course-list'][0])

    corpus_embeds = {}

    for course in corpus['Course-list']:
        corpus_embeds[course['CourseCode']] = { 'contents': model.encode(course['CourseContent']) , 'outcomes': [model.encode(outcome) for outcome in course['ILO-list-sv']] }

    # Custom JSON Encoder to handle numpy arrays
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            return super().default(obj)

    # Results in a very large file, so this will not be used
    with open('data/SU.embeddings.json', 'w') as f:
        json.dump(corpus_embeds, f, cls=NumpyEncoder)

if __name__ == '__main__':
    main()