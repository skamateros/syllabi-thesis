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

# Collect (n_courses, mean_score) per department
dept_sizes, dept_means, dept_labels = [], [], []
for dept, scores in dept_scores.items():
    dept_sizes.append(len(dept_sbert[dept]))  # number of courses
    dept_means.append(np.mean(scores))
    dept_labels.append(departments.get(dept, dept))

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
scatter = ax.scatter(dept_sizes, dept_means, alpha=0.7, 
                     s=50, color='darkslateblue')

# Label each dot with department name
for x, y, label in zip(dept_sizes, dept_means, dept_labels):
    ax.annotate(label, (x, y), fontsize=6, alpha=0.75,
                xytext=(4, 4), textcoords='offset points')

# Trend line
m, b = np.polyfit(dept_sizes, dept_means, 1)
x_line = np.linspace(min(dept_sizes), max(dept_sizes), 100)
ax.plot(x_line, m * x_line + b, color='tomato', linewidth=1.5,
        label=f'Trend (slope={m:.4f})')

ax.set_xlabel('Number of Courses')
ax.set_ylabel('Mean SBERT Similarity')
ax.set_title('Department Size vs. Mean Score')
ax.legend()
plt.tight_layout()
plt.savefig('size_vs_score.png', dpi=150)
