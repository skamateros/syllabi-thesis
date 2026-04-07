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
    abbreviations = ['t.ex.', 'd.v.s.', 'm.m.', 'bl.a.', 'etc.', 'e.g.', 'i.e.', 'ca.', 'ev.', 'fr.o.m.', 't.o.m.', 'inkl.', 'exkl.', 'o.s.v.']
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

    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    with open('data/SU.filtered.json', 'r') as f:
        corpus = json.load(f)
          
    model = SentenceTransformer('KBLab/sentence-bert-swedish-cased', device=device)

    output = {}

    match_per = 'outcome' # content or outcome

    for course in corpus['Course-list']:
        content_chunks = get_sentences(course['CourseContent'])
        content_chunk_embeds = model.encode(content_chunks, convert_to_tensor=True)

        course_outcomes = course['ILO-list-sv']
        outcome_embeds = model.encode(course_outcomes, convert_to_tensor=True)

        if match_per == 'content':
            output[course['CourseCode']] = {
                    'outcomes' : course_outcomes,
                    'chunks': []
    }
            for chunk_embed, chunk in zip(content_chunk_embeds, content_chunks):
                sim_list = [cosine_similarity(chunk_embed, outcome_embed, dim = 0).item() for outcome_embed in outcome_embeds]

                best_idx = sim_list.index(max(sim_list))
                best_sim = sim_list[best_idx]

                output[course['CourseCode']]['chunks'].append({
                    'chunk': chunk,
                    'max_similarity': best_sim,
                    'outcome_index': best_idx,
                    'matched_outcome': course_outcomes[best_idx]
                })
        elif match_per == 'outcome':
            output[course['CourseCode']] = {
                    'content_chunks' : content_chunks,
                    'outcomes': []
    }
            for outcome_embed, outcome in zip(outcome_embeds, course_outcomes):
                sim_list = [cosine_similarity(outcome_embed, chunk_embed, dim = 0).item() for chunk_embed in content_chunk_embeds]

                best_idx = sim_list.index(max(sim_list))
                best_sim = sim_list[best_idx]

                output[course['CourseCode']]['outcomes'].append({
                    'outcome': outcome,
                    'max_similarity': best_sim,
                    'chunk_index': best_idx,
                    'matched_chunk': content_chunks[best_idx]
                })
        else: 
            raise ValueError("Invalid match_per value. Use 'content' or 'outcome'.")

    with open(f'data/{match_per}.sbert.similarities.json', 'w') as f:
        json.dump(output, f, indent=2, cls=TensorEncoder)

if __name__ == '__main__':
    main()