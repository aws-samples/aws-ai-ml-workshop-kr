from mcp.server.fastmcp import FastMCP
import logging
import sys
import requests
from defusedxml import ElementTree as ET
from typing import List, Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("pubmed_mcp")

try:
    mcp = FastMCP(
        name="pubmed_tools",
    )
    logger.info("PubMed MCP server initialized successfully")
except Exception as e:
    err_msg = f"Error: {str(e)}"
    logger.error(f"{err_msg}")

# Helper functions for PubMed API
def search_pubmed(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search PubMed for articles matching the query
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        List of article dictionaries with id, title, authors, abstract, etc.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    # Search for IDs
    search_url = f"{base_url}/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance"
    }
    
    try:
        search_response = requests.get(search_url, params=search_params)
        search_response.raise_for_status()
        search_data = search_response.json()
        
        # Extract IDs
        id_list = search_data["esearchresult"]["idlist"]
        if not id_list:
            return []
            
        # Fetch article details
        fetch_url = f"{base_url}/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml"
        }
        
        fetch_response = requests.get(fetch_url, params=fetch_params)
        fetch_response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(fetch_response.text)
        articles = []
        
        for article_element in root.findall(".//PubmedArticle"):
            try:
                article = {}
                
                # Extract PMID
                pmid = article_element.find(".//PMID")
                if pmid is not None:
                    article["id"] = pmid.text
                
                # Extract title
                title = article_element.find(".//ArticleTitle")
                if title is not None:
                    article["title"] = title.text
                
                # Extract abstract
                abstract_parts = article_element.findall(".//AbstractText")
                if abstract_parts:
                    abstract = " ".join([part.text for part in abstract_parts if part.text])
                    article["abstract"] = abstract
                
                # Extract authors
                author_elements = article_element.findall(".//Author")
                if author_elements:
                    authors = []
                    for author in author_elements:
                        last_name = author.find("LastName")
                        fore_name = author.find("ForeName")
                        if last_name is not None and fore_name is not None:
                            authors.append(f"{fore_name.text} {last_name.text}")
                        elif last_name is not None:
                            authors.append(last_name.text)
                    article["authors"] = ", ".join(authors)
                
                # Extract journal info
                journal = article_element.find(".//Journal/Title")
                if journal is not None:
                    article["journal"] = journal.text
                
                # Extract publication year
                pub_date = article_element.find(".//PubDate/Year")
                if pub_date is not None:
                    article["year"] = pub_date.text
                
                articles.append(article)
            except Exception as e:
                logger.error(f"Error parsing article: {e}")
                continue
                
        return articles
    except Exception as e:
        logger.error(f"Error searching PubMed: {e}")
        return []

def get_pubmed_article_details(pmid: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific PubMed article
    
    Args:
        pmid: PubMed ID of the article
        
    Returns:
        Dictionary with article details or None if not found
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    fetch_url = f"{base_url}/efetch.fcgi"
    
    fetch_params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml"
    }
    
    try:
        fetch_response = requests.get(fetch_url, params=fetch_params)
        fetch_response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(fetch_response.text)
        article_element = root.find(".//PubmedArticle")
        
        if article_element is None:
            return None
            
        article = {
            "id": pmid,
            "references": []
        }
        
        # Extract title
        title = article_element.find(".//ArticleTitle")
        if title is not None:
            article["title"] = title.text
        
        # Extract abstract
        abstract_parts = article_element.findall(".//AbstractText")
        if abstract_parts:
            abstract = " ".join([part.text for part in abstract_parts if part.text])
            article["abstract"] = abstract
        
        # Extract authors
        author_elements = article_element.findall(".//Author")
        if author_elements:
            authors = []
            for author in author_elements:
                last_name = author.find("LastName")
                fore_name = author.find("ForeName")
                if last_name is not None and fore_name is not None:
                    authors.append(f"{fore_name.text} {last_name.text}")
                elif last_name is not None:
                    authors.append(last_name.text)
            article["authors"] = ", ".join(authors)
        
        # Extract journal info
        journal = article_element.find(".//Journal/Title")
        if journal is not None:
            article["journal"] = journal.text
        
        # Extract publication year
        pub_date = article_element.find(".//PubDate/Year")
        if pub_date is not None:
            article["year"] = pub_date.text
            
        # Extract DOI
        article_id_list = article_element.findall(".//ArticleId")
        for article_id in article_id_list:
            if article_id.get("IdType") == "doi":
                article["doi"] = article_id.text
                
        # Extract keywords
        keyword_elements = article_element.findall(".//Keyword")
        if keyword_elements:
            keywords = [k.text for k in keyword_elements if k.text]
            article["keywords"] = ", ".join(keywords)
            
        # Extract references (if available)
        reference_elements = article_element.findall(".//Reference")
        for ref in reference_elements:
            ref_data = {}
            
            # Reference citation
            citation = ref.find("Citation")
            if citation is not None:
                ref_data["citation"] = citation.text
                
            # Reference PMID (if available)
            ref_pmid = ref.find(".//ArticleId[@IdType='pubmed']")
            if ref_pmid is not None:
                ref_data["pmid"] = ref_pmid.text
                
            if ref_data:
                article["references"].append(ref_data)
                
        return article
    except Exception as e:
        logger.error(f"Error fetching article details: {e}")
        return None

# Define MCP tools
@mcp.tool()
def pubmed_search(query: str, max_results: int = 10):
    """
    Search PubMed for articles matching the query.
    
    Args:
        query: The search query for PubMed
        max_results: Maximum number of results to return (default: 10)
        
    Returns:
        List of articles with their details
    """
    logger.info(f"Searching PubMed for: {query}")
    results = search_pubmed(query, max_results)
    logger.info(f"Found {len(results)} results")
    return results

@mcp.tool()
def pubmed_get_article(pmid: str):
    """
    Get detailed information about a specific PubMed article.
    
    Args:
        pmid: PubMed ID of the article
        
    Returns:
        Detailed article information
    """
    logger.info(f"Fetching PubMed article: {pmid}")
    result = get_pubmed_article_details(pmid)
    if result:
        logger.info(f"Successfully fetched article: {pmid}")
    else:
        logger.info(f"Failed to fetch article: {pmid}")
    return result

@mcp.tool()
def pubmed_search_by_protein(protein_name: str, max_results: int = 10):
    """
    Search PubMed for articles about a specific protein.
    
    Args:
        protein_name: Name of the protein
        max_results: Maximum number of results to return (default: 10)
        
    Returns:
        List of articles about the protein
    """
    query = f"{protein_name}[Title/Abstract] AND protein[Title/Abstract]"
    logger.info(f"Searching PubMed for protein: {protein_name}")
    results = search_pubmed(query, max_results)
    logger.info(f"Found {len(results)} results for protein: {protein_name}")
    return results

@mcp.tool()
def pubmed_search_by_disease(disease_name: str, max_results: int = 10):
    """
    Search PubMed for articles about a specific disease.
    
    Args:
        disease_name: Name of the disease
        max_results: Maximum number of results to return (default: 10)
        
    Returns:
        List of articles about the disease
    """
    query = f"{disease_name}[Title/Abstract] AND (disease[Title/Abstract] OR disorder[Title/Abstract] OR condition[Title/Abstract])"
    logger.info(f"Searching PubMed for disease: {disease_name}")
    results = search_pubmed(query, max_results)
    logger.info(f"Found {len(results)} results for disease: {disease_name}")
    return results

@mcp.tool()
def pubmed_search_by_drug(drug_name: str, max_results: int = 10):
    """
    Search PubMed for articles about a specific drug.
    
    Args:
        drug_name: Name of the drug
        max_results: Maximum number of results to return (default: 10)
        
    Returns:
        List of articles about the drug
    """
    query = f"{drug_name}[Title/Abstract] AND (drug[Title/Abstract] OR medication[Title/Abstract] OR compound[Title/Abstract])"
    logger.info(f"Searching PubMed for drug: {drug_name}")
    results = search_pubmed(query, max_results)
    logger.info(f"Found {len(results)} results for drug: {drug_name}")
    return results

if __name__ == "__main__":
    mcp.run()