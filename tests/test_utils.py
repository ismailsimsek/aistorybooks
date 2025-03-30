import unittest
from llama_index.core.schema import Document
from pathlib import Path
from typing import List

from aistorybooks.utils import PdfUtil


@unittest.skip("Local user test only")
class TestPdfUtil(unittest.TestCase):

    @unittest.skip("TODO implement")
    def test_process_pdf_file_new_file(self):
        pass

    def test_split_document_into_chunks(self):
        data = [Document(text=f"Page {i}") for i in range(1, 11)]
        chunk_size = 5
        padding = 2
        skip_first_n_pages = 0

        chunks = PdfUtil.split_document_into_chunks(
            data, chunk_size, padding, skip_first_n_pages
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 7)
        self.assertEqual(len(chunks[1]), 7)

        self.assertEqual(chunks[0][0].text, "Page 1")
        self.assertEqual(chunks[0][-1].text, "Page 7")
        self.assertEqual(chunks[1][0].text, "Page 4")
        self.assertEqual(chunks[1][-1].text, "Page 10")

    def test_split_document_into_chunks_empty_data(self):
        data: List[Document] = []
        chunk_size = 5
        padding = 2
        skip_first_n_pages = 0

        chunks = PdfUtil.split_document_into_chunks(
            data, chunk_size, padding, skip_first_n_pages
        )

        self.assertEqual(len(chunks), 0)

    def test_document_info(self):
        pdf_file = Path(__file__).parent.joinpath("resources/LoremIpsum.pdf")
        document = PdfUtil.process_pdf_file(pdf_file=pdf_file, save_to_pickle=False)
        print(document[0].extra_info)
