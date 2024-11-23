from crewai import Agent, Task, Crew, Process
from langchain.chat_models import ChatOpenAI

from tools import *


class Classics2StoryBookGenerator:
    def __init__(self, book: str, author: str, language: str = "English", level: str = "A2 Beginner",
                 summary_size: str = "10 Chapter, each chapter with 100 sentences", writing_style: str = "Funny"):
        self.book = book
        self.author = author
        self.language = language
        self.level = level
        self.summary_size = summary_size
        self.writing_style = writing_style
        # self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125", max_tokens=4096, openai_api_key=Config.OPENAI_API_KEY
        #                       # openai_api_base=None
        #                       )
        # self.manager_llm = ChatOpenAI(model="gpt-3.5-turbo-0125", max_tokens=4096, openai_api_key=Config.OPENAI_API_KEY
        #                               # openai_api_base=None
        #                               )
        self.llm = ChatOpenAI(model=Config.GROQ_OPENAI_MODEL_NAME, openai_api_key=Config.GROQ_OPENAI_API_KEY,
                              openai_api_base=Config.GROQ_OPENAI_API_BASE_URL
                              )

    def create_agents(self):
        self.agent_author = Agent(
            role=f'Author',
            goal=f'Creates {self.summary_size} version of well known novels. Improves the summary with {self.writing_style} style',
            backstory=f"A creative author specialized in {self.writing_style} style stories. Improves stories with {self.writing_style} elements.",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        #
        # self.agent_critic = Agent(
        #     role=f'Literature Critic',
        #     goal=f'Reviews given story and ensures its good quality content and lengthy',
        #     backstory=f"A creative author specialized in {self.writing_style} style stories. "
        #               f"Reviews stories and ensures they are lengthy. "
        #               f"When the story is not good then asks to Author to improve it",
        #     verbose=True,
        #     llm=self.llm,
        #     allow_delegation=False
        # )

        self.agent_translator = Agent(
            role='Translator',
            goal=f"""Translates given text to {self.language} language. 
            Simplifies the translation to {self.level} level for language learners.""",
            backstory=f"An talented translator translates english text to {self.language}.",
            verbose=True,
            llm=self.llm,
            allow_delegation=False,
        )

        self.agent_image_generator = Agent(
            role='Image Generator',
            goal='Generate image for the given story. Final output should contain 1 image in json format. '
                 'Use full story text to generate the image.',
            backstory="A creative AI specialized in visual storytelling, bringing each chapter to life through "
                      "imaginative imagery.",
            verbose=True,
            llm=self.llm,
            tools=[generate_image],
            allow_delegation=False,
        )

        self.agent_content_formatter = Agent(
            role='Content Formatter',
            goal='Format the written story content in markdown, including images at the beginning of each chapter.',
            backstory='A meticulous formatter who enhances the readability and presentation of the storybook.',
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def create_tasks(self):
        self.task_summarize = Task(
            description=f'Create an summary of {self.book} from {self.author}, detailing '
                        f'a title and character descriptions. summary size should be {self.summary_size}.',
            agent=self.agent_author,
            verbose=True,
            expected_output=f'A structured outline story with the length of {self.summary_size}. '
                            f'It includes detailed character descriptions and the main plot points.'
        )
        # self.task_critic = Task(
        #     description=f'Review the story ensure it has good literature quality and smooth entertaining story. '
        #                 f'Ensure each chapter is in length of {self.summary_size}.',
        #     agent=self.agent_author,
        #     context=[self.task_summarize],
        #     verbose=True,
        #     expected_output=f'A good quality story with {self.summary_size} length.'
        # )

        self.task_generate_image = Task(
            description='Generate image that represents the story text. '
                        f'Aligning with the theme outlined in the story. '
                        f'Use the full story content to generate image. '
                        f'Dont summarize the content, use full story content.',
            agent=self.agent_image_generator,
            expected_output='A digital image file that visually represents the story.',
            context=[self.task_summarize]
        )

        self.task_translate = Task(
            description=f"""Using the story provided, translate the full story to {self.language} in {self.level} level. 
                        add repetition to it to make it better for language learners""",
            agent=self.agent_translator,
            expected_output=f"""A complete manuscript of the storybook in {self.language}. 
            Its in simple language leve land includes repetition for language learners.""",
            context=[self.task_summarize],
            verbose=True
        )

        self.task_format_content = Task(
            description='Format the story content in markdown, including an image at the beginning of the story. '
                        'Use <br> as a linebreak',
            agent=self.agent_content_formatter,
            expected_output="""The entire storybook content formatted in markdown, with imaged added at the beginning 
            of the story.""",
            context=[self.task_translate, self.task_generate_image],
            output_file="story.md"
        )

    def crew(self) -> Crew:
        crew = Crew(
            agents=[self.agent_author, self.agent_translator, self.agent_image_generator, self.agent_content_formatter],
            tasks=[self.task_summarize, self.task_translate, self.task_generate_image, self.task_format_content],
            verbose=True,
            process=Process.sequential
            # process=Process.hierarchical,
            # manager_llm = self.manager_llm
        )
        return crew

    def generate(self):
        self.create_agents()
        self.create_tasks()
        crew = self.crew()
        result = crew.kickoff()
        print(result)


if __name__ == "__main__":
    generator = Classics2StoryBookGenerator(book="The Karamazov Brothers",
                                            author="Fyodor Dostoevsky",
                                            language="German",
                                            level="A2 Beginner",
                                            summary_size="10 Chapters, each chapter more than 100 sentences log",
                                            writing_style="Philosophical")
    generator.generate()
