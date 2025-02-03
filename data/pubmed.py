import requests
import os
import time
import sys
import xml.etree.ElementTree as ET
from fake_useragent import UserAgent

def fetch_pmc_full_text(query: str, papers_metadata, max_results: int = 5) -> dict:
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    search_params = {"db": "pmc", "term": query, "retmax": max_results, "retmode": "xml"}
    search_response = requests.get(f"{base_url}/esearch.fcgi", params=search_params)
    search_response.raise_for_status()

    search_tree = ET.fromstring(search_response.content)
    pmc_ids = [id_elem.text for id_elem in search_tree.findall(".//Id")]
    print(f"pmc ids {pmc_ids}")

    if not pmc_ids:
        print("No results found.")
        return papers_metadata

    # setting up headers to not get 403'ed
    ua = UserAgent()
    header = {'User-Agent':str(ua.chrome)}

    # downloading each article as a pdf
    for pmc_id in pmc_ids:
        pdf_url = f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmc_id}/pdf/"
        destination_path = f"corpus/{pmc_id}.pdf"

        if pmc_id in papers_metadata or os.path.exists(destination_path):
            print(f"Skipping article {pmc_id}, already saved.")
            continue

        response = requests.get(pdf_url, headers=header)

        with open(destination_path, 'wb') as f:
            f.write(response.content)

        papers_metadata[pmc_id] = {"path": f"corpus/{pmc_id}.pdf"}

    return papers_metadata
