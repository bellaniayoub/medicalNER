import pandas as pd

def clean_doid_data():
    # Read the original CSV file
    df = pd.read_csv('DOID.csv')
    
    # Create a list to store all labels and synonyms
    all_terms = []
    
    # Process each row
    for _, row in df.iterrows():
        # Add the preferred label if it exists and is not empty
        if pd.notna(row['Preferred Label']) and row['Preferred Label'].strip():
            all_terms.append(row['Preferred Label'])
        
        # Add synonyms if they exist and are not empty
        if pd.notna(row['Synonyms']) and row['Synonyms'].strip():
            # Split synonyms by pipe character and add each as a separate term
            synonyms = row['Synonyms'].split('|')
            for synonym in synonyms:
                if synonym.strip():  # Only add non-empty synonyms
                    all_terms.append(synonym)
    
    # Create a new dataframe with a single column
    df_cleaned = pd.DataFrame(all_terms, columns=['Term'])
    
    # Remove duplicates
    df_cleaned = df_cleaned.drop_duplicates()
    
    # Save the cleaned data to a new CSV file
    output_path = 'DOID_cleaned.csv'
    df_cleaned.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    print(f"Original rows: {len(df)}")
    print(f"Total terms (including synonyms): {len(df_cleaned)}")

if __name__ == "__main__":
    clean_doid_data() 