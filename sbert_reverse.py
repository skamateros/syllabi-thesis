import numpy as np
import json
import re

from torch.nn.functional import cosine_similarity
import torch

from sentence_transformers import SentenceTransformer

def sliding_window(sentences, window_size=2, stride=1):
    chunks = []
    for i in range(0, len(sentences), stride):
        chunk = ". ".join(sentences[i:i+window_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def get_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

# Custom JSON Encoder to handle pytorch tensors
class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super().default(obj)
        
def main():

    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    with open('data/SU.filtered.json', 'r') as f:
        corpus = json.load(f)
          
    model = SentenceTransformer('KBLab/sentence-bert-swedish-cased', device=device)

    output = {}

    for course in corpus['Course-list']:
        sentences = get_sentences(course['CourseContent'])
        chunks = sentences
        # chunks = sliding_window(sentences, window_size=2, stride=1)
        chunk_embeds = model.encode(chunks, convert_to_tensor=True)
        outcome_embeds = model.encode(course['ILO-list-sv'], convert_to_tensor=True)
        # outcome_embeds = [model.encode(outcome, convert_to_tensor=True) for outcome in course['ILO-list-sv']]

        output[course['CourseCode']] = []

        for chunk_embed, chunk in zip(chunk_embeds, chunks):
            sim_list = [cosine_similarity(chunk_embed, outcome_embed, dim = 0).item() for outcome_embed in outcome_embeds]

            output[course['CourseCode']].append({ 'chunk': chunk,
                                            'max_similarity': max(sim_list),
                                            'outcome_index': sim_list.index(max(sim_list)),
                                            'outcomes': course['ILO-list-sv']})
        
        # output[course['CourseCode']] = [[cosine_similarity(chunk_embed, outcome_embed, dim = 0).item() for chunk_embed in chunk_embeds] for outcome_embed in outcome_embeds]
        # output[course['CourseCode']] = [[sim] for sim in max_similarities]

    with open('data/SU.reverse.sbert.similarities.json', 'w') as f:
        json.dump(output, f, indent=2, cls=TensorEncoder)

if __name__ == '__main__':
    main()