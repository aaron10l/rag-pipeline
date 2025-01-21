import os
import requests
from bs4 import BeautifulSoup

API_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
SEARCH_URL = API_BASE_URL + "esearch.fcgi"
FETCH_URL = API_BASE_URL + "efetch.fcgi"
OUTPUT_DIR = "./text/"
NUM_ARTICLES = 5
TERM = "protein synthesis"

def search_pubmed(term, num_articles):
    params = {
        "db": "pmc",
        "term": term,
        "retmax": num_articles,
        "sort": "pubdate",
        "retmode": "json",
    }
    response = requests.get(SEARCH_URL, params=params)
    response.raise_for_status()
    return response.json()

def fetch_full_article_xml(pmc_id):
    params = {
        "db": "pmc",
        "id": pmc_id,
        "rettype": "xml",
        "retmode": "text",
    }
    response = requests.get(FETCH_URL, params=params)
    response.raise_for_status()
    print(response.text)
    return response.text

def extract_text_from_xml(xml_content):
    soup = BeautifulSoup(xml_content, "lxml")
    body = soup.find("body")
    if not body:
        return "Full text not available in XML."
    paragraphs = body.find_all("p")
    return "\n\n".join(p.get_text() for p in paragraphs)

def save_article(pmc_id, content):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    file_path = os.path.join(OUTPUT_DIR, f"{pmc_id}.txt")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

def main():
    print("Searching for articles...")
    # search_results = search_pubmed(TERM, NUM_ARTICLES)
    # pmc_ids = search_results.get("esearchresult", {}).get("idlist", [])

    pmc_ids = ["PMC10760240", "PMC6789920","PMC8096971", "PMC1820505", "PMC2248754"]

    if not pmc_ids:
        print("No articles found.")
        return

    print(f"Found {len(pmc_ids)} articles. Fetching and saving...")
    for pmc_id in pmc_ids:
        try:
            print(f"Fetching full article XML for {pmc_id}...")
            xml_content = fetch_full_article_xml(pmc_id)
            full_text = extract_text_from_xml(xml_content)
            save_article(pmc_id, full_text)
            print(f"Saved article {pmc_id}.")
        except Exception as e:
            print(f"Failed to fetch or save article {pmc_id}: {e}")

    print(f"Articles saved in '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main()
