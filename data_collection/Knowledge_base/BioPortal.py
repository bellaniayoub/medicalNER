import requests

# Replace with your BioPortal API key
API_KEY = "4f89d918-aa1c-4be3-a80c-3af6794447aa"
ONTOLOGY_ACRONYM = "GO"  # Example: Gene Ontology (publicly accessible)

# BioPortal API endpoints
ONTOLOGY_DOWNLOAD_URL = f"https://data.bioontology.org/ontologies/{ONTOLOGY_ACRONYM}/download"
ONTOLOGY_METADATA_URL = f"https://data.bioontology.org/ontologies/{ONTOLOGY_ACRONYM}"

# Headers required by BioPortal API
headers = {
    "Authorization": f"apikey token={API_KEY}",
    "Accept": "application/json"
}

def download_ontology():
    try:
        # Step 1: Verify ontology accessibility
        metadata_response = requests.get(ONTOLOGY_METADATA_URL, headers=headers)
        
        if metadata_response.status_code == 200:
            print(f"Ontology '{ONTOLOGY_ACRONYM}' exists and is accessible.")
            
            # Step 2: Download the ontology
            download_response = requests.get(ONTOLOGY_DOWNLOAD_URL, headers=headers)
            
            if download_response.status_code == 200:
                # Save to file
                with open(f"{ONTOLOGY_ACRONYM}.owl", "wb") as f:
                    f.write(download_response.content)
                print(f"Downloaded {ONTOLOGY_ACRONYM}.owl successfully!")
            else:
                print(f"Download failed (HTTP {download_response.status_code}): {download_response.text}")
        
        elif metadata_response.status_code == 403:
            print("403 Forbidden: Check your API key or ontology permissions.")
        elif metadata_response.status_code == 404:
            print(f"Ontology '{ONTOLOGY_ACRONYM}' not found.")
        else:
            print(f"Unexpected error: {metadata_response.status_code}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    download_ontology()