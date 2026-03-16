def filter_corpus(corpus):
    red_flags = ['praktik', 'projekt', 'project', 'thesis', 'uppsats', 'examensarbete', 'självständigt arbete']
    corpus['Course-list'] = [
        course for course in corpus['Course-list']
        if not any(word in course['CourseTitle'].lower() for word in red_flags)
        and len(course['ILO-list-sv']) <= 10
        and len(course['ILO-list-sv']) > 0
    ]
    return corpus

def main():
    import json

    with open('data/SU.heuristics.json', 'r') as f:
        corpus = json.load(f)

    print('Before', len(corpus['Course-list']))

    corpus = filter_corpus(corpus)

    print('After', len(corpus['Course-list']))

    with open('data/SU.filtered.json', 'w') as f:
        json.dump(corpus, f, indent=2)

if __name__ == '__main__':
    main()