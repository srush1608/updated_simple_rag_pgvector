
import os
from PyPDF2 import PdfReader

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, "rb") as file:
            reader = PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text.replace("\x00", "")
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def load_text_samples(file_path, is_pdf=False):
    texts = []
    if is_pdf:
        texts.append(extract_text_from_pdf(file_path))
    else:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file '{file_path}' not found.")
        with open(file_path, 'r') as file:
            texts = list(dict.fromkeys(filter(None, file.read().splitlines())))
    return texts
