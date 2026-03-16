import json
import matplotlib.pyplot as plt
import numpy as np

with open('../data/SU.departments.json', 'r') as f:
    departments = json.load(f)
with open('../data/SU.hybrid.similarities.json', 'r') as f:
    sbert = json.load(f)
with open('../data/SU.filtered.json', 'r') as f:
    corpus = json.load(f)

course_deps = {}
for course in corpus['Course-list']:
    course_deps[course['CourseCode']] = course['Department']

dept_sbert = {dep: [] for dep in departments.keys()}
for course_code in sbert:
    dept = course_deps[course_code]
    if dept not in dept_sbert:
        continue
    dept_sbert[dept].append(sbert[course_code])

# Collect scores per department
dept_scores = {}
for dept in dept_sbert:
    sims = []
    n = 0
    for course_sims in dept_sbert[dept]:
        sims.extend(sim[0] for sim in course_sims)
        n += len(course_sims)
    if n == 0:
        continue
    dept_scores[dept] = sims  # store all scores, not just mean

# Build display data
labels = [departments.get(d, d) for d in dept_scores]
means  = [np.mean(scores) for scores in dept_scores.values()]
errors = [np.std(scores) for scores in dept_scores.values()]

# Sort by mean descending
sorted_idx = np.argsort(means)
labels  = [labels[i] for i in sorted_idx]
means   = [means[i]  for i in sorted_idx]
errors  = [errors[i] for i in sorted_idx]

# Plot
fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.35)))

bars = ax.barh(labels, means, xerr=errors, 
               color='darkslateblue', ecolor='gray',
               capsize=3, alpha=0.85)

# Mean threshold line
grand_mean = np.mean(means)
ax.axvline(grand_mean, color='tomato', linestyle='--', linewidth=1.2,
           label=f'Mean: {grand_mean:.3f}')

ax.set_xlabel('Average SBERT Similarity')
ax.set_title('Per-Department SBERT Similarity Scores')
ax.legend()
ax.set_xlim(left=0)
plt.tight_layout()
plt.savefig('dept_similarity.png', dpi=150)
