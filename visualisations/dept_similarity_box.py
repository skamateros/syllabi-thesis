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

# Sort departments by median score
sorted_items = sorted(dept_scores.items(), key=lambda x: np.median(x[1]))
sorted_labels = [departments.get(d, d) for d, _ in sorted_items]
sorted_data   = [scores for _, scores in sorted_items]

# Plot
fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_labels) * 0.35)))

bp = ax.boxplot(sorted_data, vert=False, patch_artist=True,
                medianprops=dict(color='tomato', linewidth=1.5),
                boxprops=dict(facecolor='darkslateblue', alpha=0.7),
                flierprops=dict(marker='o', markersize=3, 
                                color='gray', alpha=0.5))

ax.set_yticks(range(1, len(sorted_labels) + 1))
ax.set_yticklabels(sorted_labels)

# Grand median line
grand_median = np.median([s for scores in sorted_data for s in scores])
ax.axvline(grand_median, color='tomato', linestyle='--', linewidth=1.2,
           label=f'Overall median: {grand_median:.3f}')

ax.set_xlabel('SBERT Similarity')
ax.set_title('Per-Department SBERT Similarity Distributions')
ax.legend()
plt.tight_layout()
plt.savefig('dept_similarity_box.png', dpi=150)
