import os
from typing import List, Optional

os.environ["OTEL_SDK_DISABLED"] = "true"
from crewai import Agent, Task, Crew, Process
from crewai import LLM
from crewai.agents.agent_builder.base_agent import BaseAgent

from config import Config  # Assuming you have a config.py file
from tools import ImageGenerator


class StoryBookGenerator:
    """
    A class to generate a storybook based on a classic novel, including translation,
    image generation, and content formatting.
    """

    def __init__(
            self,
            book: str,
            author: str,
            language: str = "English",
            level: str = "A2 Beginner",
            summary_size: str = "10 Chapter, each chapter with 100 sentences",
            writing_style: str = "Funny",
    ):
        """
        Initializes the StoryBookGenerator with book details and preferences.

        Args:
            book (str): The title of the classic novel.
            author (str): The author of the classic novel.
            language (str, optional): The target language for translation. Defaults to "English".
            level (str, optional): The language proficiency level for translation. Defaults to "A2 Beginner".
            summary_size (str, optional): The desired size of the summary. Defaults to "10 Chapter, each chapter with 100 sentences".
            writing_style (str, optional): The writing style for the summary. Defaults to "Funny".
        """
        self.book = book
        self.author = author
        self.language = language
        self.level = level
        self.summary_size = summary_size
        self.writing_style = writing_style
        self.llm = self._initialize_llm()
        self._tasks: Optional[List[Task]] = None
        self._crew: Optional[Crew] = None

    def _initialize_llm(self) -> LLM:
        """Initializes the language model."""
        return LLM(
            model="openai/" + Config.OPENAI_MODEL_NAME,
            api_key=Config.OPENAI_API_KEY,
            api_base=Config.OPENAI_API_BASE_URL,
        )
        # return LLM(
        #     model="groq/" + Config.GROQ_OPENAI_MODEL_NAME,
        #     api_key=Config.GROQ_OPENAI_API_KEY,
        #     api_base=Config.GROQ_OPENAI_API_BASE_URL,
        # )

    def _create_author_agent(self) -> Agent:
        """Creates the Author agent."""
        return Agent(
            role="Author",
            goal=f"Creates {self.summary_size} version of well known novels. Improves the summary with {self.writing_style} style",
            backstory=f"A creative author specialized in {self.writing_style} style stories. Improves stories with {self.writing_style} elements.",
            verbose=True,
            llm=self.llm,
            allow_delegation=False,
        )

    def _create_translator_agent(self) -> Agent:
        """Creates the Translator agent."""
        return Agent(
            role="Translator",
            goal=f"""Translates given text to {self.language} language. 
            Simplifies the translation to {self.level} level for language learners.""",
            backstory=f"An talented translator translates english text to {self.language}.",
            verbose=True,
            llm=self.llm,
            allow_delegation=False,
        )

    def _create_image_generator_agent(self) -> Agent:
        """Creates the Image Generator agent."""
        return Agent(
            role="Image Generator",
            goal="Generate image for the given story. Final output should contain 1 image in json format. "
                 "Use full story text to generate the image.",
            backstory="A creative AI specialized in visual storytelling, bringing each chapter to life through "
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
            goal="Format the written story content in markdown, including images at the beginning of each chapter.",
            backstory="A meticulous formatter who enhances the readability and presentation of the storybook.",
            verbose=True,
            llm=self.llm,
            allow_delegation=False,
        )

    def _create_summarize_task(self) -> Task:
        """Creates the Summarize task."""
        author_agent = self._create_author_agent()
        return Task(
            description=f"Create an summary of {self.book} from {self.author}, detailing "
                        f"a title and character descriptions. summary size should be {self.summary_size}.",
            agent=author_agent,
            expected_output=f"A structured outline story with the length of {self.summary_size}. "
                            f"It includes detailed character descriptions and the main plot points.",
        )

    def _create_generate_image_task(self, summarize_task: Task) -> Task:
        """Creates the Generate Image task."""
        image_generator_agent = self._create_image_generator_agent()
        return Task(
            description="Generate image that represents the story text. "
                        f"Aligning with the theme outlined in the story. "
                        f"Use the full story content to generate image. "
                        f"Dont summarize the content, use full story content.",
            agent=image_generator_agent,
            expected_output="A digital image file that visually represents the story.",
            context=[summarize_task],
        )

    def _create_translate_task(self, summarize_task: Task) -> Task:
        """Creates the Translate task."""
        translator_agent = self._create_translator_agent()
        return Task(
            description=f"""Using the story provided, translate the full story to {self.language} in {self.level} level. 
                        add repetition to it to make it better for language learners""",
            agent=translator_agent,
            expected_output=f"""A complete manuscript of the storybook in {self.language}. 
            Its in simple language leve land includes repetition for language learners.""",
            context=[summarize_task],
        )

    def _create_format_content_task(self, translate_task: Task, generate_image_task: Task) -> Task:
        """Creates the Format Content task."""
        content_formatter_agent = self._create_content_formatter_agent()
        return Task(
            description="Format the story content in markdown, including an image at the beginning of the story. "
                        "Use <br> as a linebreak",
            agent=content_formatter_agent,
            expected_output="""The entire storybook content formatted in markdown, with imaged added at the beginning 
            of the story.""",
            context=[translate_task, generate_image_task],
            output_file="story.md",
        )

    @property
    def tasks(self) -> List[Task]:
        """Creates all the tasks required for the storybook generation."""
        if self._tasks is None:
            summarize_task = self._create_summarize_task()
            generate_image_task = self._create_generate_image_task(summarize_task)
            translate_task = self._create_translate_task(summarize_task)
            format_content_task = self._create_format_content_task(translate_task, generate_image_task)

            self._tasks = [summarize_task, translate_task, generate_image_task, format_content_task]
        return self._tasks

    @property
    def agents(self) -> List[BaseAgent]:
        """Creates all the agents required for the storybook generation."""
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
        """Generates the storybook by creating agents, tasks, and running the crew."""
        result = self.crew.kickoff()
        print(result)


if __name__ == "__main__":
    generator = StoryBookGenerator(
        book="The Karamazov Brothers",
        author="Fyodor Dostoevsky",
        language="German",
        level="A2 Beginner",
        summary_size="10 Chapters, each chapter more than 100 sentences log",
        writing_style="Philosophical",
    )
    generator.generate()
