import scrapy
import json
from datetime import datetime
from scrapy.utils.log import logger
import csv

class PubmedSpider(scrapy.Spider):
    name = 'pubmed_spider'
    allowed_domains = ['pubmed.ncbi.nlm.nih.gov']
    
    def __init__(self):
        # Create CSV file for output
        self.csv_file = open('free_pmc_articles.csv', 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Search Term', 'PMID', 'URL', 'Title', 'Authors', 'Journal', 'Year'])
        
        # Define search terms
        # self.search_terms = [
        #     'cancer',
        #     'diabetes',
        #     'hypertension',
        #     'asthma',
        #     'cardiovascular disease',
        #     'stroke',
        #     'alzheimer',
        #     'parkinson',
        #     'depression',
        #     'anxiety',
        #     'arthritis',
        #     'osteoporosis',
        #     'multiple sclerosis',
        #     'epilepsy',
        #     'autism',
        #     'schizophrenia',
        #     'bipolar disorder',
        #     'hiv',
        #     'hepatitis',
        #     'tuberculosis',
        #     'malaria',
        #     'dengue',
        #     'covid-19',
        #     'influenza',
        #     'pneumonia',
        #     'chronic kidney disease',
        #     'liver disease',
        #     'gastrointestinal disorders',
        #     'inflammatory bowel disease',
        #     'rheumatoid arthritis'
        # ]
        self.search_terms = [
        'cancer',
        'diabetes',
        'hypertension',
        'asthma',
        'cardiovascular disease',
        'stroke',
        'alzheimer',
        'parkinson',
        'depression',
        'anxiety',
        'arthritis',
        'osteoporosis',
        'multiple sclerosis',
        'epilepsy',
        'autism',
        'schizophrenia',
        'bipolar disorder',
        'hiv',
        'hepatitis',
        'tuberculosis',
        'malaria',
        'dengue',
        'covid-19',
        'influenza',
        'pneumonia',
        'chronic kidney disease',
        'liver disease',
        'gastrointestinal disorders',
        'inflammatory bowel disease',
        'rheumatoid arthritis',
        'eczema',
        'psoriasis',
        'lupus',
        'celiac disease',
        'crohn\'s disease',
        'ulcerative colitis',
        'anemia',
        'leukemia',
        'lymphoma',
        'melanoma',
        'pancreatitis',
        'hepatitis b',
        'hepatitis c',
        'obesity',
        'metabolic syndrome',
        'sepsis',
        'meningitis',
        'encephalitis',
        'migraine',
        'glaucoma',
        'cataract',
        'macular degeneration',
        'retinopathy',
        'otitis media',
        'sinusitis',
        'bronchitis',
        'emphysema',
        'copd',
        'sleep apnea',
        'insomnia',
        'narcolepsy',
        'endometriosis',
        'polycystic ovary syndrome',
        'prostate cancer',
        'testicular cancer',
        'breast cancer',
        'ovarian cancer',
        'cervical cancer',
        'colon cancer',
        'rectal cancer',
        'skin cancer',
        'thyroid disease',
        'goiter',
        'graves\' disease',
        'hashimoto\'s thyroiditis',
        'addison\'s disease',
        'cushing\'s syndrome',
        'pituitary tumor',
        'acromegaly',
        'gigantism',
        'dwarfism',
        'sickle cell anemia',
        'thalassemia',
        'hemophilia',
        'deep vein thrombosis',
        'pulmonary embolism',
        'myocardial infarction',
        'angina',
        'arrhythmia',
        'atrial fibrillation',
        'ventricular tachycardia',
        'bradycardia',
        'heart failure',
        'cardiomyopathy',
        'pericarditis',
        'endocarditis',
        'aortic aneurysm',
        'varicose veins',
        'hemorrhoids',
        'gallstones',
        'kidney stones',
        'urinary tract infection',
        'bladder cancer',
        'renal failure',
        'dialysis'
    ]

    def start_requests(self):
        # Generate requests for each search term
        for term in self.search_terms:
            # URL encode the search term
            encoded_term = term.replace(' ', '+')
            base_url = f'https://pubmed.ncbi.nlm.nih.gov/?term={encoded_term}'
            logger.info(f"Starting search for term: {term}")
            yield scrapy.Request(
                url=base_url,
                callback=self.parse_search_results,
                meta={'search_term': term}
            )
    
    def parse_search_results(self, response):
        search_term = response.meta['search_term']
        logger.info(f"Processing search results for term: {search_term}")
        
        # Extract article divs
        article_divs = response.css('div.docsum-wrap')
        logger.info(f"Found {len(article_divs)} articles for term: {search_term}")
        
        for article in article_divs:
            # Check if article has "Free PMC article" citation
            free_pmc = article.css('span.free-resources.spaced-citation-item.citation-part::text').get()
            if free_pmc and "Free PMC article" in free_pmc:
                # Extract PMID
                pmid = article.css('span.docsum-pmid::text').get()
                
                # Extract article link
                article_link = article.css('a.docsum-title::attr(href)').get()
                full_url = f"https://pubmed.ncbi.nlm.nih.gov{article_link}"
                
                # Extract title
                title = article.css('a.docsum-title::text').get().strip()
                
                # Extract authors
                authors = article.css('span.docsum-authors.full-authors::text').get()
                
                # Extract journal and year
                journal_citation = article.css('span.docsum-journal-citation.full-journal-citation::text').get()
                year = journal_citation.split(';')[0].strip() if journal_citation else ''
                
                # Write to CSV
                self.csv_writer.writerow([search_term, pmid, full_url, title, authors, journal_citation, year])
                logger.info(f"Found free PMC article for {search_term}: {title} (PMID: {pmid})")
        
        # Follow pagination
        next_page = response.css('a.next-page-button::attr(href)').get()
        if next_page:
            logger.info(f"Found next page for {search_term}: {next_page}")
            yield response.follow(
                next_page,
                callback=self.parse_search_results,
                meta={'search_term': search_term}
            )
        else:
            logger.info(f"No more pages found for {search_term}")
    
    def closed(self, reason):
        # Close CSV file when spider is done
        self.csv_file.close()
        logger.info("CSV file closed")