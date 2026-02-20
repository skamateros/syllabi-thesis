import numpy as np
import json

from torch.nn.functional import cosine_similarity
import torch

from sentence_transformers import SentenceTransformer

def main():
    with open('data/SU.heuristics.json', 'r') as f:
        corpus = json.load(f)
    
    model = SentenceTransformer('KBLab/sentence-bert-swedish-cased')

    output = {}

    for course in corpus['Course-list']:
        content_embed = model.encode(course['CourseContent'])
        outcome_embeds = [model.encode(outcome) for outcome in course['ILO-list-sv']]
        
        output[course['CourseCode']] = [cosine_similarity(torch.as_tensor(content_embed), torch.as_tensor(outcome_embed), dim = 0) for outcome_embed in outcome_embeds]        

    # Custom JSON Encoder to handle pytorch tensors
    class TensorEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            return super().default(obj)

    with open('data/SU.similarities.json', 'w') as f:
        json.dump(output, f, indent=4, cls=TensorEncoder)

if __name__ == '__main__':
    main()