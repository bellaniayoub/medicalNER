import csv
import os
import requests
import time
from scrapy.utils.log import logger
import re

class PMCPDFSpider:
    def __init__(self):
        # Create output directory
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'pmc_pdfs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Base URL for E-utilities API
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
        # Headers to mimic browser request
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Read URLs from the previous spider's CSV output
        self.articles_to_process = []
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'free_pmc_articles.csv')
        
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.articles_to_process.append({
                        'pmid': row['PMID'],
                        'title': row['Title']
                    })
            logger.info(f"Loaded {len(self.articles_to_process)} articles to process")
        else:
            logger.error(f"CSV file not found: {csv_path}")
    
    def get_pmc_id(self, pmid):
        """Get PMC ID for a given PMID using E-utilities API"""
        url = f"{self.base_url}/esearch.fcgi"
        params = {
            'db': 'pmc',
            'term': f"{pmid}[pmid]",
            'retmode': 'json'
        }
        
        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if 'esearchresult' in data and 'idlist' in data['esearchresult']:
                pmc_ids = data['esearchresult']['idlist']
                if pmc_ids:
                    return pmc_ids[0]
            return None
        except Exception as e:
            logger.error(f"Error getting PMC ID for PMID {pmid}: {str(e)}")
            return None
    
    def get_pdf_url(self, pmc_id):
        """Get PDF URL for a given PMC ID"""
        # The correct format for PMC PDF URLs
        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/main.pdf"
    
    def download_pdf(self, pdf_url, filepath):
        """Download PDF from URL"""
        try:
            # First try to get the PDF page
            response = requests.get(pdf_url, headers=self.headers, allow_redirects=True)
            response.raise_for_status()
            
            # Check if we got a PDF
            if 'application/pdf' in response.headers.get('Content-Type', ''):
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                logger.error(f"URL did not return a PDF: {pdf_url}")
                return False
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                logger.error(f"Access forbidden (403) for URL: {pdf_url}")
                logger.info("This article might not be available for direct PDF download.")
            else:
                logger.error(f"HTTP error downloading PDF: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error downloading PDF from {pdf_url}: {str(e)}")
            return False
    
    def process_articles(self):
        """Process all articles from the CSV file"""
        for article in self.articles_to_process:
            pmid = article['pmid']
            title = article['title']
            
            logger.info(f"Processing article: {title} (PMID: {pmid})")
            
            # Get PMC ID
            pmc_id = self.get_pmc_id(pmid)
            if not pmc_id:
                logger.warning(f"No PMC ID found for PMID {pmid}")
                continue
            
            # Get PDF URL
            pdf_url = self.get_pdf_url(pmc_id)
            if not pdf_url:
                logger.warning(f"No PDF URL found for PMC ID {pmc_id}")
                continue
            
            # Clean the title to make it a valid filename
            clean_title = re.sub(r'[<>:"/\\|?*]', '_', title)
            clean_title = clean_title[:100]  # Limit length
            
            # Create filename
            filename = f"{pmid}_{clean_title}.pdf"
            filepath = os.path.join(self.output_dir, filename)
            
            # Download PDF
            if self.download_pdf(pdf_url, filepath):
                logger.info(f"Successfully downloaded PDF for article: {title}")
            else:
                logger.error(f"Failed to download PDF for article: {title}")
            
            # Respect rate limits (3 requests per second)
            time.sleep(0.34)  # Slightly more than 1/3 second to be safe

if __name__ == "__main__":
    spider = PMCPDFSpider()
    spider.process_articles() 