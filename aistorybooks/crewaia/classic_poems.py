import os
from typing import List, Optional

os.environ["OTEL_SDK_DISABLED"] = "true"
from crewai import Agent, Task, Crew, Process
from crewai import LLM
from crewai.agents.agent_builder.base_agent import BaseAgent

from config import Config  # Assuming you have a config.py file
from tools import ImageGenerator, MarkdownToPdfConverter


class ClassicPoemGenerator:
    """
    A class to generate a poem based on a classic novel, including image generation,
    and content formatting.
    """

    def __init__(
            self,
            book: str,
            author: str,
            poetic_style: str = "Philosophical",
    ):
        """
        Initializes the ClassicPoemGenerator with book details and preferences.

        Args:
            book (str): The title of the classic novel.
            author (str): The author of the classic novel.
            poetic_style (str, optional): The poetic style for the poem. Defaults to "Philosophical".
        """
        self.book = book
        self.author = author
        self.poetic_style = poetic_style
        self.llm = self._initialize_llm()
        self._tasks: Optional[List[Task]] = None
        self._crew: Optional[Crew] = None

    def _initialize_llm(self) -> LLM:
        """Initializes the language model."""
        return LLM(
            model="groq/" + Config.GROQ_MODEL_NAME,
            api_key=Config.GROQ_API_KEY,
            api_base=Config.GROQ_API_BASE_URL,
        )

    def _create_poet_agent(self) -> Agent:
        """Creates the Poet agent."""
        return Agent(
            role="Poet",
            goal=f"Turns given story, novel or literature content to a beautiful poem.",
            backstory=f"An talented poet who summarizes novels and turns them to a beautiful, imaginary and "
                      f"metaphorical poems.",
            verbose=True,
            llm=self.llm,
            allow_delegation=False,
        )

    def _create_creative_poet_agent(self) -> Agent:
        """Creates the Creative Poet agent."""
        return Agent(
            role="Creative Poet",
            goal=f"Takes the given poem and improves it with {self.poetic_style} style",
            backstory=f"A creative author specialized in {self.poetic_style} style poems. "
                      f"Improves poems with {self.poetic_style} elements.",
            verbose=True,
            llm=self.llm,
            allow_delegation=False,
        )

    def _create_image_generator_agent(self) -> Agent:
        """Creates the Image Generator agent."""
        return Agent(
            role="Image Generator",
            goal="Generate one image for the poem written by the creative poet. Final output should "
                 "contain 1 image in json format. Use full poem text to generate the image.",
            backstory="A creative AI specialized in visual storytelling, bringing each poem to life through "
                      "imaginative imagery.",
            verbose=True,
            llm=self.llm,
            tools=[ImageGenerator()],
            allow_delegation=False,
        )

    def _create_content_formatter_agent(self) -> Agent:
        """Creates the Content Formatter agent."""
        return Agent(
            role="Content Formatter",
            goal="Format the written story content in markdown, including image at the beginning of each poem.",
            backstory="A meticulous formatter who enhances the readability and presentation of the storybook.",
            verbose=True,
            llm=self.llm,
            allow_delegation=False,
        )

    def _create_markdown_to_pdf_creator_agent(self) -> Agent:
        """Creates the PDF Converter agent."""
        return Agent(
            role="PDF Converter",
            goal="Convert the Markdown file to a PDF document. poem.md is the markdown file name.",
            backstory="An efficient converter that transforms Markdown files into professionally formatted PDF "
                      "documents.",
            verbose=True,
            llm=self.llm,
            tools=[MarkdownToPdfConverter()],
            allow_delegation=False,
        )

    def _create_create_poem_task(self) -> Task:
        """Creates the Create Poem task."""
        poet_agent = self._create_poet_agent()
        return Task(
            description=f"Create an summary of {self.book} from {self.author} as a poem. "
                        f"Giving it a title and character descriptions and making it {self.poetic_style} style poem. "
                        f"Include title of the poem at the top.",
            agent=poet_agent,
            expected_output=f"A structured document with the poem and title of the poem at the top.",
        )

    def _create_improve_poem_task(self, create_poem_task: Task) -> Task:
        """Creates the Improve Poem task."""
        creative_poet_agent = self._create_creative_poet_agent()
        return Task(
            description=f"Improve the given poem with {self.poetic_style} style. ",
            agent=creative_poet_agent,
            expected_output=f"A improved poem with {self.poetic_style} style.",
            context=[create_poem_task],
        )

    def _create_generate_image_task(self, improve_poem_task: Task) -> Task:
        """Creates the Generate Image task."""
        image_generator_agent = self._create_image_generator_agent()
        return Task(
            description="Generate 1 image that represents the poem text. "
                        f"Aligning with the {self.poetic_style} theme outlined in the poem. "
                        f"Use the full poem content to generate image. Dont summarize it, ise it as is.",
            agent=image_generator_agent,
            expected_output="A digital image file that visually represents the poem.",
            context=[improve_poem_task],
        )

    def _create_format_content_task(self, improve_poem_task: Task, generate_image_task: Task) -> Task:
        """Creates the Format Content task."""
        content_formatter_agent = self._create_content_formatter_agent()
        return Task(
            description="Format given poem content as a markdown document, "
                        "Including an image after the title and before the poem. Use <br> as a linebreak",
            agent=content_formatter_agent,
            expected_output="""The entire poem content formatted in markdown, With linebreak after each verse. image 
            added after the title""",
            context=[improve_poem_task, generate_image_task],
            output_file="poem.md",
        )

    def _create_markdown_to_pdf_task(self, format_content_task: Task) -> Task:
        """Creates the Markdown to PDF task."""
        markdown_to_pdf_creator_agent = self._create_markdown_to_pdf_creator_agent()
        return Task(
            description="Convert a Markdown file to a PDF document, ensuring the preservation of formatting, "
                        "structure, and embedded images using the mdpdf library.",
            agent=markdown_to_pdf_creator_agent,
            expected_output="A PDF file generated from the Markdown input, accurately reflecting the content with "
                            "proper formatting. The PDF should be ready for sharing or printing.",
            context=[format_content_task],
        )

    @property
    def tasks(self) -> List[Task]:
        """Creates all the tasks required for the poem generation."""
        if self._tasks is None:
            create_poem_task = self._create_create_poem_task()
            improve_poem_task = self._create_improve_poem_task(create_poem_task)
            generate_image_task = self._create_generate_image_task(improve_poem_task)
            format_content_task = self._create_format_content_task(improve_poem_task, generate_image_task)
            markdown_to_pdf_task = self._create_markdown_to_pdf_task(format_content_task)

            self._tasks = [create_poem_task, improve_poem_task, generate_image_task, format_content_task,
                           markdown_to_pdf_task]
        return self._tasks

    @property
    def agents(self) -> List[BaseAgent]:
        """Creates all the agents required for the poem generation."""
        return [task.agent for task in self.tasks]

    @property
    def crew(self) -> Crew:
        """Creates the Crew with agents and tasks."""
        if self._crew is None:
            self._crew = Crew(
                agents=self.agents,
                tasks=self.tasks,
                verbose=True,
                process=Process.sequential,
            )
        return self._crew

    def generate(self):
        """Generates the poem by creating agents, tasks, and running the crew."""
        result = self.crew.kickoff()
        print(result)


if __name__ == "__main__":
    generator = ClassicPoemGenerator(
        book="The Karamazov Brothers",
        author="Fyodor Dostoevsky",
        poetic_style="Alexander Pushkin and Philosophical",
    )
    generator.generate()
