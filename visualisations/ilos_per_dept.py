import json
import numpy as np
import matplotlib.pyplot as plt

with open('../data/SU.departments.json', 'r') as f:
    departments = json.load(f)
with open('../data/SU.filtered.json', 'r') as f:
    corpus = json.load(f)

# Collect ILO counts per department BEFORE filtering
dept_ilos = {}
for course in corpus['Course-list']:
    dept = departments.get(course['Department'], course['Department'])
    ilo_count = len(course['ILO-list-sv'])
    dept_ilos.setdefault(dept, []).append(ilo_count)

# Sort by median
sorted_items = sorted(dept_ilos.items(), key=lambda x: np.median(x[1]))
sorted_labels = [f"{label} (n={len(data)})" for label, data in sorted_items]
sorted_data   = [data  for _, data  in sorted_items]

# Plot
fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_labels) * 0.35)))

ax.boxplot(sorted_data, vert=False, patch_artist=True,
           medianprops=dict(color='tomato', linewidth=1.5),
           boxprops=dict(facecolor='darkslateblue', alpha=0.7),
           flierprops=dict(marker='o', markersize=3, color='gray', alpha=0.5))

ax.set_yticks(range(1, len(sorted_labels) + 1))
ax.set_yticklabels(sorted_labels)

grand_median = np.median([n for data in sorted_data for n in data])
ax.axvline(grand_median, color='tomato', linestyle='--', linewidth=1.2,
           label=f'Overall median: {grand_median:.1f}')

ax.set_xlabel('Number of ILOs')
ax.set_title('ILOs per Course per Department')
plt.figtext(0.99, 0.003, 'Value in parentheses indicates number of courses in department',
            horizontalalignment='right', fontsize=8, color='gray')
ax.legend()
plt.tight_layout()
plt.savefig('ilos_per_dept.png', dpi=150)
