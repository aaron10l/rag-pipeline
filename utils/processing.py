import glob

def process_text_files(folder_path: str, chunk_size: int = 512) -> dict:
    file_paths = glob.glob(f"{folder_path}/*.txt")
    return {file_path: load_and_chunk_text(file_path, chunk_size) for file_path in file_paths}

def load_and_chunk_text(file_path: str, chunk_size: int = 512, stride: int = 256) -> list:
    with open(file_path, 'r', encoding="utf-8") as file:
        text = file.read()
    return [text[i:i + chunk_size] for i in range(0, len(text) - chunk_size + 1, stride)]
