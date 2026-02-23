import json

deps = {}
with open('data/SU.departments.txt', 'rt') as f:
    for line in f:
        code, name = line.strip().split(' ', 1)
        deps[code] = name

with open('data/SU.departments.json', 'w') as f:
    json.dump(deps, f, indent=2)