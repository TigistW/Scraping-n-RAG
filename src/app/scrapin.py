import streamlit as st
import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

# Helper functions
def search_pubmed(keyword, retmax=10):
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
        st.error(f"Error resolving DOI: {e}")
        return None

def find_pdf_link(article_url, base_url):
    try:
        response = requests.get(article_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        pdf_link = soup.find('a', href=lambda href: href and href.endswith('.pdf'))

        if pdf_link:
            pdf_href = pdf_link['href']
            pdf_url = urljoin(base_url, pdf_href)
            return pdf_url
    except Exception as e:
        st.error(f"Error fetching PDF link :{e}")
    return None

def download_pdf(pdf_url, output_filename):
    try:
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        with open(output_filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        return True
    except Exception as e:
        st.error(f"Error:{e}")
        return False

def clean_doi(doi_id):
    if doi_id.startswith("doi: "):
        doi_id = doi_id.replace("doi: ", "").strip()
    return f"https://doi.org/{doi_id}"

# Streamlit UI
st.title("PubMed Article Search and PDF Downloader")

# Input fields
keyword = st.text_input("Enter a keyword to search PubMed:", "")
max_results = st.number_input("Number of articles to retrieve:", min_value=1, max_value=50, value=10, step=1)

if st.button("Search PubMed"):
    if keyword:
        with st.spinner("Searching PubMed..."):
            article_ids = search_pubmed(keyword, max_results)
            if article_ids:
                st.success(f"Found {len(article_ids)} articles.")
                articles = []
                for pmid in article_ids:
                    details = fetch_article_details(pmid)
                    article_title = details.get("title", "No Title Available")
                    article_url = details.get("elocationid", "")
                    article_url = clean_doi(article_url) if article_url else None
                    articles.append({
                        "PMID": pmid,
                        "Title": article_title,
                        "DOI URL": article_url
                    })
                st.session_state.articles = articles
            else:
                st.warning("No articles found for the given keyword.")
    else:
        st.error("Please enter a keyword to search.")

# Display results
if "articles" in st.session_state and st.session_state.articles:
    st.subheader("Search Results")
    for article in st.session_state.articles:
        cols = st.columns([3, 1])  # Create two columns: one for the DOI and one for the button
        with cols[0]:
            st.write(f"**DOI URL:** {article['DOI URL'] or 'N/A'}")
        with cols[1]:
            if article["DOI URL"]:
                base_url = resolve_doi_to_base_url(article["DOI URL"])
                pdf_link = find_pdf_link(article["DOI URL"], base_url)
                if pdf_link:
                    filename = f"{article['PMID']}.pdf"
                    if st.button(f"Download PDF", key=article['PMID']):
                        with st.spinner(f"Downloading PDF: {article['Title']}..."):
                            if download_pdf(pdf_link, filename):
                                st.success(f"PDF downloaded: {filename}")
                                st.write(f"Downloaded: {filename}")
                else:
                    st.warning("No PDF link found.")
            else:
                st.warning("No DOI available.")
