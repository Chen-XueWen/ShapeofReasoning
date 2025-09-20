from bs4 import BeautifulSoup, NavigableString
import os
import pandas as pd

def extract_text(element):
    # Recursively extract text and alt attributes from images
    parts = []
    for child in element.descendants:
        if isinstance(child, NavigableString):
            parts.append(child.string)
        elif child.name == 'img':
            alt = child.get('alt', '')
            if alt:
                parts.append(alt)
    return ''.join(parts)

def parse_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # Extract problem
    problem = ''
    heading = soup.find(id='Problem')
    if heading:
        lines = []
        for sib in heading.parent.next_siblings:
            if sib.name == 'h2':
                break
            if sib.name:
                txt = extract_text(sib)
                if txt:
                    lines.append(txt.strip())
        problem = '\n'.join(lines)

    # Extract solutions
    solutions = []
    for span in soup.find_all('span', class_='mw-headline'):
        ident = span.get('id')
        if ident and ident.startswith('Solution'):
            h2 = span.parent
            lines = []
            for sib in h2.next_siblings:
                if sib.name == 'h2':
                    break
                if sib.name:
                    txt = extract_text(sib)
                    if txt:
                        lines.append(txt.strip())
            sol_text = '\n'.join(lines)
            solutions.append(f"{span.get_text().strip()}:\n{sol_text}")
    combined_solutions = '\n\n'.join(solutions)

    return problem, combined_solutions

# Directory containing the HTML files
dirpath = './2024_aime_problems'

# List to store records
records = []
for filename in sorted(os.listdir(dirpath)):
    if filename.endswith('.html'):
        filepath = os.path.join(dirpath, filename)
        problem, solutions = parse_file(filepath)
        records.append({'filename': filename, 'problem': problem, 'solutions': solutions})

# Create a DataFrame and save to Excel
df = pd.DataFrame(records)
output = './aime_problems_extracted_2024.xlsx'
df.to_excel(output, index=False)

print(f'Saved to {output}')