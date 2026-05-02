import json

with open('../data/lowest_alignment_sbert.json', 'r') as f:
    lowest_alignment = json.load(f)

def save():
    with open('../data/lowest_alignment_sbert.json', 'w') as f:
        json.dump(lowest_alignment, f, indent=4)

categories = { '1': 'Pre-processing error',
               '2': 'True Misalignment',
               '3': 'Unidentified Alignment',
               '4': 'Ambiguous' }

def main():
    for direction in ["lowest_ilos_to_content", "lowest_content_to_ilos"]:
        print(f"Annotating {direction}...")
        for item in lowest_alignment[direction]:
            if not item['category']:
                if direction == "lowest_content_to_ilos":
                    print(f"Course: {item['course_code']}")
                    print(f"Department: {item['department']}")
                    print(f"Chunk: {item['chunk']}")
                    print(f"Matched Outcome: {item['matched_outcome']}")
                    print(f"Similarity: {item['similarity']}")
                    print("-" * 40)

                elif direction == "lowest_ilos_to_content":
                    print(f"Course: {item['course_code']}")
                    print(f"Department: {item['department']}")
                    print(f"Outcome: {item['outcome']}")
                    print(f"Matched Chunk: {item['matched_chunk']}")
                    print(f"Similarity: {item['similarity']}")
                    print("-" * 40)
                
                while True:
                    category = input("Enter category (1-4): ")
                    if category in ['1', '2', '3', '4']:
                        item['category'] = categories[category]
                        break
                    elif category == 'exit':
                        print("Exiting annotation process.")
                        save()
                        exit()
                    else:
                        print("Invalid category. Please enter one of the specified options.")

if __name__ == "__main__":
    main()