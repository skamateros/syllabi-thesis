import numpy as np
import json

from torch.nn.functional import cosine_similarity
import torch

from sentence_transformers import SentenceTransformer

def main():

    use_chunking = False

    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    with open('data/SU.heuristics.json', 'r') as f:
    # with open('data/SU.subset.json', 'r') as f:
        corpus = json.load(f)
    
    model = SentenceTransformer('KBLab/sentence-bert-swedish-cased', device=device)

    output = {}

    for course in corpus['Course-list']:
        if use_chunking:
            chunks = course['CourseContent'].split('. ')
        else:
            chunks = [course['CourseContent']]

        chunk_embeds = [model.encode(chunk, convert_to_tensor=True) for chunk in chunks]
        outcome_embeds = [model.encode(outcome, convert_to_tensor=True) for outcome in course['ILO-list-sv']]

        output[course['CourseCode']] = [[cosine_similarity(chunk_embed, outcome_embed, dim = 0).item() for chunk_embed in chunk_embeds] for outcome_embed in outcome_embeds]

    # Custom JSON Encoder to handle pytorch tensors
    class TensorEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            return super().default(obj)

    with open('data/SU.sbert.similarities.json', 'w') as f:
        json.dump(output, f, indent=2, cls=TensorEncoder)

if __name__ == '__main__':
    main()