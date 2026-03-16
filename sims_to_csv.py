import json
from statistics import mean
import pandas as pd

with open('data/SU.departments.json', 'r') as f:
    departments = json.load(f)
with open('data/SU.filtered.json', 'r') as f:
    corpus = json.load(f)
with open('data/SU.hybrid.similarities.json', 'r') as f:
    sbert = json.load(f)

df = pd.DataFrame(columns = ['CourseCode', 'Department', 'SBERT_Similarity'])

for course in corpus['Course-list']:
    code = course['CourseCode']
    dept = departments.get(course['Department'])
    sims = [s[0] for s in sbert.get(code)]
    num_ilos = len(course['ILO-list-sv'])
    if sims == []:
        print(code, sims)
        continue
    sim = mean(sims)

    datapoint = pd.DataFrame({'CourseCode': code, 'Department': dept, 'ILO_Count': num_ilos, 'SBERT_Similarity': sim}, index=[0])
    df = pd.concat([df, datapoint], ignore_index=True)

# print(df.head())

df.to_csv('data/course_similarities.csv', index=False)