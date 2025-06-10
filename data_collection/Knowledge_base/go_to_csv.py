import csv

def extract_go_terms(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['GO_ID', 'Name'])  # CSV header

        term = {}
        for line in f_in:
            line = line.strip()
            if line.startswith('[Term]'):
                term = {}  # Reset for new term
            elif line.startswith('id:'):
                term['id'] = line.split('id: ')[1].strip()
            elif line.startswith('name:'):
                term['name'] = line.split('name: ')[1].strip()
            elif line == '' and term:
                # Write to CSV if the term is not obsolete
                if 'is_obsolete' not in term:
                    writer.writerow([term.get('id', ''), term.get('name', '')])
                term = {}  # Reset term

# Example usage
input_file = 'GO.owl'  # Replace with your input file
output_file = 'go_terms.csv'
extract_go_terms(input_file, output_file)