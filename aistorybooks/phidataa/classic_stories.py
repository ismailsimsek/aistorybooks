import httpx
from openai._base_client import SyncHttpxClientWrapper
from pathlib import Path
from phi.agent import Agent as PhiAgent
from phi.model.google.gemini import Gemini
from phi.model.openai.like import OpenAILike
from phi.utils.log import logger as PhiLogger
from phi.workflow import RunResponse, RunEvent
from typing import Optional, Union, Iterator

from aistorybooks.config import Config
from aistorybooks.utils import PdfUtil


class OpenAILikeNoVerifySSL(OpenAILike):

    def __init__(self, base_url: Optional[Union[str, httpx.URL]] = None, *args, **kwargs):
        http_client = SyncHttpxClientWrapper(
            base_url=base_url,
            verify=False
        )
        super().__init__(http_client=http_client, base_url=base_url, *args, **kwargs)


class PhiStoryBookGenerator:

    def __init__(
            self,
            language: str = "English",
            level: str = "A2 Beginner",
            summary_size: str = "approximately 100 sentences and approximately 800 hundred words",
            writing_style: str = "Funny",
            **kwargs
    ):
        self.language = language
        self.level = level
        self.summary_size = summary_size
        self.writing_style = writing_style
        self.model = Gemini(
            id=Config.GEMINI_MODEL_NAME,
            api_key=Config.GEMINI_API_KEY
        )

        self.author_agent = PhiAgent(
            model=self.model,
            description="Expert author rewriting novels to a shortened stories.",
            task=(
                f"Rewrite given the novel text to {self.summary_size} a story, "
                f"rewrite it in a {self.writing_style} style. "
                f"Target {self.level} language learners. Repeat key words. Maintain narrative flow."
                f"This is a first section of the big novel. Next sections will follow."
            ),
            markdown=True,
            debug_mode=True,
        )

        self.translator_agent = PhiAgent(
            model=self.model,
            description=f"Expert English to {self.language} translator.",
            task=(
                f"Translate Given English text to {self.language} for {self.level} level language learners. "
                f"Repeat key words. Ensure natural story flow."
            ),
            markdown=True,
            debug_mode=True,
        )

    def return_if_response_none(self, response: RunResponse):
        if response is None:
            yield RunResponse(event=RunEvent.workflow_completed, content=f"Sorry, received empty result")

    # @TODO add tenacity and retry api call "rate limit exceptions"!
    def _run_chunk(self, content: str, start_page, end_page) -> RunResponse:
        try:
            summary: RunResponse = self.author_agent.run(content)
            if summary is None or summary.content is None:
                return RunResponse(event="RunFailed",
                                   content=f"Failed to generate summary for pages {start_page}-{end_page}")

            translated: RunResponse = self.translator_agent.run(summary.content)
            if translated is None or translated.content is None:
                return RunResponse(event="RunFailed",
                                   content=f"Failed to translate summary for pages {start_page}-{end_page}")
            return translated
        except Exception as e:
            return RunResponse(event="RunFailed", content=f"Error processing pages {start_page}-{end_page}: {e}")

    def run(self, pdf_file: Path, chunk_size=10, padding=1, skip_first_n_pages=0) -> Iterator[RunResponse]:
        # final = pdf_file.parent.joinpath(f"{pdf_file.stem}.md")
        data = PdfUtil.process_pdf_file(pdf_file)
        chunks = PdfUtil.split_document_into_chunks(
            data=data,
            chunk_size=chunk_size,
            padding=padding,
            skip_first_n_pages=skip_first_n_pages
        )

        total_chunks = len(chunks)
        for index, chunk in enumerate(chunks):
            start_page = chunk[0].extra_info.get('page')
            end_page = chunk[-1].extra_info.get('page')
            total_pages = chunk[-1].extra_info.get('total_pages')
            PhiLogger.info(f"Processing Pages: {start_page}-{end_page}")
            chunk_str = "\n\n".join([doc.text for doc in chunk])

            response = self._run_chunk(content=chunk_str, start_page=start_page, end_page=end_page)

            if response.event != "RunFailed":
                response.metrics[
                    'progress_info'] = f"Processed Pages: {skip_first_n_pages}-{end_page} of {total_pages}"
                response.metrics['progress_total'] = total_chunks
                response.metrics['progress_current_index'] = index + 1
                response.metrics['progress_percent'] = int(((index + 1) / total_chunks) * 100)
            yield response
