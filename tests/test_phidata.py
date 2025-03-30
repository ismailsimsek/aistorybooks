import unittest
from pathlib import Path

from aistorybooks.phidata.classic_stories import PhiStoryBookGenerator


@unittest.skip("Local user test only")
class TestPhiStoryBookGenerator(unittest.TestCase):

    def test_story_book_generator(self):
        pdf_file = Path(__file__).parent.joinpath("resources/LoremIpsum.pdf")

        generator = PhiStoryBookGenerator(
            language="German",
            level="A1 Intermediate",
            summary_size="Long (150 sentences/1200 words)",
            writing_style="Philosophical",
        )

        results = generator.run(pdf_file=pdf_file, chunk_size=1, padding=0, skip_first_n_pages=0)

        i = 0
        for result in results:
            i += 1
            print(result.metrics)
            if i > 2:
                raise "STOP"

    @unittest.skip("Local user test only")
    def test_story_book_generator_run_chunk(self):
        generator = PhiStoryBookGenerator(
            language="German",
            level="A1 Intermediate",
            summary_size="Long (150 sentences/1200 words)",
            writing_style="Funny",
        )

        result = generator.author_agent.run("hello, how are you doing")
        print(result)
        result = generator._run_chunk(content="hello, how are you doing", start_page=0, end_page=1)
        print(result)
