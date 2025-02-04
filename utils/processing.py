import glob
import os
from grobid_client.grobid_client import GrobidClient
import xml.etree.ElementTree as ET

def process_text_files(folder_path: str, chunk_size: int = 512) -> dict:
    pdf_paths = glob.glob(f"{folder_path}/*.pdf")
    return {file_path: load_and_chunk_text(file_path, chunk_size) for file_path in pdf_paths}

def load_and_chunk_text(file_path: str, chunk_size: int = 512, stride: int = 256) -> list:
    with open(file_path, 'r', encoding="utf-8") as file:
        text = file.read()
    return [text[i:i + chunk_size] for i in range(0, len(text) - chunk_size + 1, stride)]

def convert_pdf_to_xml(folder_path, grobid_config='./config.json'):
    output_path = './xml_texts'
    if any(os.listdir(output_path)):
        return output_path
    client = GrobidClient(config_path=grobid_config)
    client.process("processFulltextDocument", folder_path, output=output_path, n=1)
    print("all pdf files converted to xml and saved in ")
    return output_path

def convert_xml_to_txt(folder_path):
    xml_paths = glob.glob(f"{folder_path}/*.xml")
    output_dir= './txt_texts'
    os.makedirs(output_dir, exist_ok=True)

    for path in xml_paths:
        tree = ET.parse(path)
        root = tree.getroot()
        text = []

        # Extract namespace dynamically
        ns = {'tei': root.tag.split("}")[0].strip("{")} if '}' in root.tag else {}

        # Find the title using the extracted namespace
        title_elem = root.find(".//tei:titleStmt/tei:title", ns)
        text.append(f"{title_elem.text}\n")

        # Extract body text
        body_elem = root.find(".//tei:text/tei:body", ns)
        
        # Extract all paragraph texts within the body
        if body_elem is not None:
            for paragraph in body_elem.findall(".//tei:p", ns):  # Extract all <p> elements inside <body>
                if paragraph.text:
                    text.append(paragraph.text.strip())

        document_text = ".".join(text)

        path_split = path.split("/")
        document_number = path_split[-1].split(".", 2)[0]
        with open(f"{output_dir}/{document_number}.txt", "w", encoding='utf-8') as f:
            f.write(document_text)

    print(f"all .txt files saved to {output_dir}")
    
