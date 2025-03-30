import pickle
import pymupdf4llm
from llama_index.core.schema import Document
from pathlib import Path
from typing import List


class PdfUtil:

    @staticmethod
    def process_pdf_file(pdf_file: Path, save_to_pickle: bool = False) -> List[Document]:
        pickle_file = pdf_file.parent.joinpath(f"{pdf_file.stem}.pkl")

        if pickle_file.exists() and save_to_pickle is True:
            print(f"Loading data from pickle file: {pickle_file}")
            with open(pickle_file, "rb") as f:
                data = pickle.load(f)
        else:
            print(f"Processing PDF file: {pdf_file}")
            reader = pymupdf4llm.LlamaMarkdownReader()
            data: List[Document] = reader.load_data(pdf_file)

            if save_to_pickle:
                print(f"Saving data to pickle file: {pickle_file}")
                with open(pickle_file, "wb") as f:
                    pickle.dump(data, f)

        return data

    @staticmethod
    def split_document_into_chunks(data: List[Document], chunk_size: int, padding: int, skip_first_n_pages=0) -> List[
        List[Document]]:
        chunks: List[List[Document]] = []
        for i in range(skip_first_n_pages, len(data), chunk_size):
            start_index = max(skip_first_n_pages, i - padding)
            end_index = min(len(data), i + chunk_size + padding)
            chunk_pages = data[start_index:end_index]
            chunks.append(chunk_pages)
        return chunks
