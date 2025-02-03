import requests
import time
import sys
import xml.etree.ElementTree as ET

def fetch_pmc_full_text(query: str, papers_metadata, max_results: int = 5) -> dict:
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    search_params = {"db": "pmc", "term": query, "retmax": max_results, "retmode": "xml"}
    search_response = requests.get(f"{base_url}/esearch.fcgi", params=search_params)
    search_response.raise_for_status()

    search_tree = ET.fromstring(search_response.content)
    pmc_ids = [id_elem.text for id_elem in search_tree.findall(".//Id")]

    if not pmc_ids:
        print("No results found.")
        return papers_metadata

    fetch_url = f"{base_url}/efetch.fcgi"

    for pmc_id in pmc_ids:
        if pmc_id in papers_metadata:
            print(f"Skipping article {pmc_id}, already saved.")
            continue

        fetch_params = {"db": "pmc", "id": pmc_id, "rettype": "full", "retmode": "xml"}
        fetch_response = requests.get(fetch_url, params=fetch_params)
        fetch_response.raise_for_status()
        time.sleep(1)

        article_tree = ET.fromstring(fetch_response.content)
        body_elements = article_tree.findall(".//body")

        if not body_elements:
            print(f"No full text body found for article {pmc_id}.")
            continue

        full_text = "\n".join(ET.tostring(body, encoding="unicode", method="text") for body in body_elements)
        file_name = f"corpus/PMC_{pmc_id}.txt"

        with open(file_name, "w", encoding="utf-8") as file:
            file.write(full_text)

        papers_metadata[pmc_id] = {"file_name": file_name}
        print(f"Saved full text of article {pmc_id} as {file_name}")

    return papers_metadata
