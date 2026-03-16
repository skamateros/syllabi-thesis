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

# Count courses per department from the filtered corpus
dept_counts = {}
for course in corpus['Course-list']:
    dept = course['Department']
    label = departments.get(dept, dept)
    dept_counts[label] = dept_counts.get(label, 0) + 1

# Sort by count
labels, counts = zip(*sorted(dept_counts.items(), key=lambda x: x[1]))

# Plot
fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.35)))
ax.barh(labels, counts, color='darkslateblue', alpha=0.85)
for i, count in enumerate(counts):
    ax.text(count + 0.2, i, str(count), va='center', fontsize=8)
ax.set_xlim(right=max(counts) * 1.1)

mean_count = np.mean(counts)
ax.axvline(mean_count, color='tomato', linestyle='--', linewidth=1.2,
           label=f'Mean: {mean_count:.1f}')

ax.set_xlabel('Number of Courses')
ax.set_title('Courses per Department (filtered)')
ax.legend()
plt.tight_layout()
plt.savefig('courses_per_dept.png', dpi=150)
