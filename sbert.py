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
    abbreviations = ['t.ex.', 'd.v.s.', 'm.m.', 'bl.a.', 'etc.']
    placeholder_map = {}
    for i, abbr in enumerate(abbreviations):
        placeholder = f"__ABBR_{i}__"
        placeholder_map[placeholder] = abbr
        text = text.replace(abbr, placeholder)

    sentences = re.split(r'(?<!\b[a-zA-Z]\.)(?<!\b\d\.)(?<=[.!?])\s+', text)

    restored = []
    for sentence in sentences:
        for placeholder, abbr in placeholder_map.items():
            sentence = sentence.replace(placeholder, abbr)
        restored.append(sentence.strip())

    return [s for s in restored if s]

# Custom JSON Encoder to handle pytorch tensors
class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super().default(obj)
        
def main():

    use_chunking = True

    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    with open('data/SU.filtered.json', 'r') as f:
        corpus = json.load(f)
          
    model = SentenceTransformer('KBLab/sentence-bert-swedish-cased', device=device)

    output = {}

    for course in corpus['Course-list']:
        if use_chunking:
            sentences = get_sentences(course['CourseContent'])
            chunks = sliding_window(sentences, window_size=2, stride=1)
        else:
            chunks = [course['CourseContent']]

        chunk_embeds = model.encode(chunks, convert_to_tensor=True)
        outcome_embeds = model.encode(course['ILO-list-sv'], convert_to_tensor=True)
        # outcome_embeds = [model.encode(outcome, convert_to_tensor=True) for outcome in course['ILO-list-sv']]

        max_similarities = [torch.max(cosine_similarity(chunk_embeds, outcome_embed.unsqueeze(0), dim=1)).item()
                    for outcome_embed in outcome_embeds]
        
        # output[course['CourseCode']] = [[cosine_similarity(chunk_embed, outcome_embed, dim = 0).item() for chunk_embed in chunk_embeds] for outcome_embed in outcome_embeds]
        output[course['CourseCode']] = [[sim] for sim in max_similarities]

    with open('data/SU.chunked.similarities.json', 'w') as f:
        json.dump(output, f, indent=2, cls=TensorEncoder)

if __name__ == '__main__':
    main()