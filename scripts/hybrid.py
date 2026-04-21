# Run as such:
# python3 hybrid.py {sim_file1} {sim_file2} {optional: output_file}

import correlation
import json
from sys import argv

def hybridize(file1: str, file2: str, output_file: str = None, alpha: float = 0.5) -> dict:
    with open(file1, 'r') as f1:
        sim_json1 = json.load(f1)

    with open(file2, 'r') as f2:
        sim_json2 = json.load(f2)
    
    print(f"Hybridizing with alpha = {alpha}")
    print(f"sim = {file1.split('.')[1]} + {alpha} * {file2.split('.')[1]}")
    new_sim = {}
    for course_code in sim_json1:
        if course_code in sim_json2:
            # [( alpha * sim1 + (1-alpha) * sim2) / 2]
            # [ sim1 + alpha * sim2 ]
            new_sim[course_code] = [[ sim1 + alpha * sim2 ] for sim1, sim2 in zip([max(sim) for sim in sim_json1[course_code]], [max(sim) for sim in sim_json2[course_code]])]
    
    if output_file is not None:
        with open(output_file, 'w') as f4:
            json.dump(new_sim, f4, indent = 2)
    
    return new_sim

def main():

    if len(argv) < 3:
        print("Usage: python3 hybrid.py {sim_file1} {sim_file2} {optional: output_file}")
        exit(1)
    if len(argv) == 4:
        new_sim = hybridize(argv[1], argv[2], argv[3])
    else:
        for a in range(0,10,1):
            a = a / 10
            new_sim = hybridize(argv[1], argv[2], alpha = a)

            correlation.correlation(f_similarities=new_sim, f_corpus='data/SU.subset.json', method=correlation.spearmanr)

if __name__ == '__main__':
    main()