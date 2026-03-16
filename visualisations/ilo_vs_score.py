import json
import matplotlib.pyplot as plt
import numpy as np

with open('../data/SU.departments.json', 'r') as f:
    departments = json.load(f)
with open('../data/SU.hybrid.similarities.json', 'r') as f:
    sbert = json.load(f)
with open('../data/SU.filtered.json', 'r') as f:
    corpus = json.load(f)

# Build a lookup for ILO count per course from the (unfiltered) corpus
ilo_counts = {}
for course in corpus['Course-list']:
    ilo_counts[course['CourseCode']] = len(course['ILO-list-sv'])

# Collect (ilo_count, score) pairs
ilo_x, score_y = [], []
for course_code, course_sims in sbert.items():
    if course_code not in ilo_counts:
        continue
    avg_score = np.mean([sim[0] for sim in course_sims])
    ilo_x.append(ilo_counts[course_code])
    score_y.append(avg_score)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(ilo_x, score_y, alpha=0.4, s=15, color='darkslateblue')

# Trend line — filter out any NaN pairs before fitting
ilo_arr = np.array(ilo_x)
score_arr = np.array(score_y)
mask = np.isfinite(ilo_arr) & np.isfinite(score_arr)
m, b = np.polyfit(ilo_arr[mask], score_arr[mask], 1)
x_line = np.linspace(ilo_arr[mask].min(), ilo_arr[mask].max(), 100)
ax.plot(x_line, m * x_line + b, color='tomato', linewidth=1.5,
        label=f'Trend (slope={m:.4f})')

ax.set_xlabel('Number of ILOs')
ax.set_ylabel('Average SBERT Similarity')
ax.set_title('Score vs. ILO Count per Course')
ax.legend()
plt.tight_layout()
plt.savefig('ilo_vs_score.png', dpi=150)
