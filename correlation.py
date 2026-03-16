# Run as such:
# python3 correlation.py {pearsonr/spearmanr} {similarities_file} {corpus_file}

# python3 correlation.py spearmanr data/SU.similarities.json data/SU.subset.json

from sys import argv
from scipy.stats import spearmanr, pearsonr
import json

def correlation(f_similarities, f_corpus, method = spearmanr):
    print(f"Calculating {method.__name__} correlation...")
    x1 = []
    x2 = []

    if isinstance(f_similarities, str):
        with open(f_similarities, 'r') as f1:
            similarities = json.load(f1)
    elif isinstance(f_similarities, dict):
        similarities = f_similarities
    if isinstance(f_corpus, str):
        with open(f_corpus, 'r') as f2:
            corpus = json.load(f2)
    elif isinstance(f_corpus, dict):
        corpus = f_corpus
        
    for course in corpus['Course-list']:
        course_code = course['CourseCode']
        if course_code in similarities:
            for outcome in similarities[course_code]:
                if isinstance(outcome, list):
                    x1.append(max(outcome))
                else:
                    x1.append(outcome)
            x2.extend(course["ILO-relation"])

    assert len(x1) == len(x2), "The two lists must have the same length"

    result = method(x1, x2)

    print(f"Correlation coefficient: {result.correlation:.4f}; P-value: {result.pvalue:.4f}")

def main():
    if len(argv) != 4:
        print("Usage: python3 correlation.py {pearsonr/spearmanr} {similarities_file} {corpus_file}")
        exit(1)
    if argv[1] == "pearsonr":
        method = pearsonr
    elif argv[1] == "spearmanr":
        method = spearmanr
    else:
        print("Invalid method. Use 'pearsonr' or 'spearmanr'.")
        exit(1)
    correlation(method=method, f_similarities=argv[2], f_corpus=argv[3])

if __name__ == '__main__':
    main()