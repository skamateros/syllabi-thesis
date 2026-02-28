# Run as such:
# python3 correlation.py {pearsonr/spearmanr} {similarities_file} {corpus_file}

# python3 correlation.py pearsonr data/SU.similarities.json data/SU.subset.json

from sys import argv
from scipy.stats import spearmanr, pearsonr
import json

def correlation(f_similarities:str, f_corpus:str, method = pearsonr):
    print(f"Calculating {method.__name__} correlation...")
    x1 = []
    x2 = []
    with open(f_similarities, 'r') as f1:
        similarities = json.load(f1)
        with open(f_corpus, 'r') as f2:
            corpus = json.load(f2)
            for course in corpus['Course-list']:
                course_code = course['CourseCode']
                if course_code in similarities:
                    x1.extend(similarities[course_code])
                    x2.extend(course["ILO-relation"])

    assert len(x1) == len(x2), "The two lists must have the same length"

    result = method(x1, x2)

    print(f"Correlation coefficient: {result.correlation:.4f}")
    print(f"P-value: {result.pvalue:.4f}")

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