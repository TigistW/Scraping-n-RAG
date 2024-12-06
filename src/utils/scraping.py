import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

def search_pubmed(keyword, retmax=10):
    # Search PubMed for articles related to a keyword
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": keyword,
        "retmax": retmax,
        "retmode": "json"
    }
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    data = response.json()
    print("data", data)
    return data.get("esearchresult", {}).get("idlist", [])

def fetch_article_details(pmid):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "json"
    }
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    data = response.json()
    return data.get("result", {}).get(pmid, {})

def resolve_doi_to_base_url(doi_url):
    try:
        response = requests.get(doi_url, allow_redirects=True)
        response.raise_for_status()
        resolved_url = response.url
        parsed_url = urlparse(resolved_url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        return base_url
    except Exception as e:
        print(f"Error resolving DOI: {e}")
        return None

def find_pdf_link(article_url, base_url):
    try:
        response = requests.get(article_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        pdf_link = soup.find('a', href=lambda href: href and href.endswith('.pdf'))
        print("pdf_link tag:", pdf_link)

        if pdf_link:
            pdf_href = pdf_link['href']
            pdf_url = urljoin(base_url, pdf_href)
            return pdf_url
    except Exception as e:
        print(f"Error fetching PDF link: {e}")
    return None

def download_pdf(pdf_url, output_filename):
    # Download the PDF from the given URL.
    try:
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        with open(output_filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"PDF downloaded: {output_filename}")
    except Exception as e:
        print(f"Error downloading PDF: {e}")

def clean_doi(doi_id):
    if doi_id.startswith("doi: "):
        doi_id = doi_id.replace("doi: ", "").strip()
    return f"https://doi.org/{doi_id}"

def search_google_scholar(query, max_result):
    try:
        search_results = scholarly.search_pubs(query)
        for i, article in enumerate(search_results):
            title = article['bib']['title']
            doi = article.get('doi', None)
            pdf_url = article.get('eprint_url', None)
            print(f"\nTitle: {title}")
            print(f"DOI: {doi}")
            print(f"Direct PDF URL: {pdf_url}")

            # Attempt to fetch the PDF
            if pdf_url:
                download_pdf(pdf_url, f"{title}.pdf")
            elif doi:
                fetch_pdf_from_doi(doi)
            else:
                print("No direct PDF or DOI found.")
            if i == max_result:
                break
    except Exception as e:
        print(f"Error fetching Google Scholar results: {e}")

def fetch_pdf_from_doi(doi):
    try:
        doi_url = f"https://doi.org/{doi}"
        print(f"Resolving DOI: {doi_url}")
        # Resolve the DOI to get the actual article URL
        response = requests.get(doi_url, allow_redirects=True)
        response.raise_for_status()
        resolved_url = response.url
        print(f"Resolved URL: {resolved_url}")

        # Fetch the PDF link from the resolved article URL
        pdf_url = find_pdf_link(resolved_url)
        if pdf_url:
            download_pdf(pdf_url, f"{doi.replace('/', '_')}.pdf")
        else:
            print("No PDF link found on the article page.")
    except Exception as e:
        print(f"Error resolving DOI: {e}")

# def main():
keyword = input("Enter a keyword for PubMed search: ")
max_num = 15
article_ids = search_pubmed(keyword, max_num)
print(f"Found {len(article_ids)} articles.")
for pmid in article_ids:
    details = fetch_article_details(pmid)
    article_title = details.get("title", "No Title")
    article_url = details.get("elocationid", "")
    print(f"PMID: {pmid}, Title: {article_title}")
    article_url = clean_doi(article_url)
    base_url = resolve_doi_to_base_url(article_url)
    print("DOI", article_url)
    if article_url:
        pdf_link = find_pdf_link(article_url, base_url)
        if pdf_link:
            filename = f"{pmid}.pdf"
            download_pdf(pdf_link, filename)
        else:
            print("No PDF link found.")
