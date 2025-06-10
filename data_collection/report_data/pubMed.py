import requests
import xml.etree.ElementTree as ET
import time
import pandas as pd

def fetch_pubmed_abstracts(search_term, max_results=100):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    # First get the list of PMIDs
    search_url = f"{base_url}esearch.fcgi?db=pubmed&term={search_term}&retmax={max_results}&usehistory=y"
    search_response = requests.get(search_url)
    search_root = ET.fromstring(search_response.content)
    
    # Extract query key and web environment for fetching details
    web_env = search_root.find(".//WebEnv").text
    query_key = search_root.find(".//QueryKey").text
    count = int(search_root.find(".//Count").text)
    
    print(f"Found {count} results")
    
    # Set up for batch fetching
    batch_size = 20
    abstracts = []
    
    # Fetch in batches
    for start in range(0, min(count, max_results), batch_size):
        time.sleep(1)  # Be nice to the server
        fetch_url = f"{base_url}efetch.fcgi?db=pubmed&query_key={query_key}&WebEnv={web_env}&retstart={start}&retmax={batch_size}&retmode=xml"
        fetch_response = requests.get(fetch_url)
        
        # Parse the XML
        root = ET.fromstring(fetch_response.content)
        articles = root.findall(".//PubmedArticle")
        
        for article in articles:
            try:
                pmid = article.find(".//PMID").text
                title_element = article.find(".//ArticleTitle")
                title = title_element.text if title_element is not None else "No title"
                
                abstract_text_elements = article.findall(".//AbstractText")
                abstract = " ".join([elem.text for elem in abstract_text_elements if elem.text]) if abstract_text_elements else "No abstract"
                
                abstracts.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract
                })
                
            except Exception as e:
                print(f"Error processing article: {e}")
    
    return pd.DataFrame(abstracts)

# Example usage
df = fetch_pubmed_abstracts("medical+reports+clinical", max_results=200)
df.to_csv("pubmed_medical_reports.csv", index=False)