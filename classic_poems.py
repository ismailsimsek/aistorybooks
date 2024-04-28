from crewai import Agent, Task, Crew, Process
from langchain.chat_models import ChatOpenAI

from tools import *

os.environ["OPENAI_API_BASE"] = Config.GROQ_OPENAI_API_BASE
os.environ["OPENAI_API_KEY"] = Config.GROQ_OPENAI_API_KEY


class Classics2PoemGenerator:
    def __init__(self, book: str, author: str, poetic_style: str = "Philosophical"):
        self.book = book
        self.author = author
        self.poetic_style = poetic_style
        self.llm = ChatOpenAI(model=Config.GROQ_OPENAI_MODEL_NAME)

    def create_agents(self):
        self.agent_poet = Agent(
            role='Poet',
            goal=f"""Turns given story, novel or literature content to a beautiful poem.""",
            backstory=f"An talented poet who summarizes novels and turns them to a beautiful, imaginary and "
                      f"metaphorical poems.",
            verbose=True,
            llm=self.llm,
            allow_delegation=False,
        )

        self.agent_creative_poet = Agent(
            role=f'Creative Poet',
            goal=f'Takes the given poem and improves it with {self.poetic_style} style',
            backstory=f"A creative author specialized in {self.poetic_style} style poems. "
                      f"Improves poems with {self.poetic_style} elements.",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

        self.agent_image_generator = Agent(
            role='Image Generator',
            goal='Generate one image for the poem writen by the creative poet. Final output should '
                 'contain 1 image in json format. Use full poem text to generate the image.',
            backstory="A creative AI specialized in visual storytelling, bringing each poem to life through "
                      "imaginative imagery.",
            verbose=True,
            llm=self.llm,
            tools=[generate_image],
            allow_delegation=False,
        )

        self.agent_content_formatter = Agent(
            role='Content Formatter',
            goal='Format the written story content in markdown, including image at the beginning of each poem.',
            backstory='A meticulous formatter who enhances the readability and presentation of the storybook.',
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

        self.agent_markdown_to_pdf_creator = Agent(
            role='PDF Converter',
            goal='Convert the Markdown file to a PDF document. poem.md is the markdown file name.',
            backstory='An efficient converter that transforms Markdown files into professionally formatted PDF '
                      'documents.',
            verbose=True,
            llm=self.llm,
            tools=[convert_markdown_to_pdf],
            allow_delegation=False
        )

    def create_tasks(self):
        self.task_create_poem = Task(
            description=f'Create an summary of {self.book} from {self.author} as a poem. '
                        f'Giving it a title and character descriptions and making it {self.poetic_style} style poem. '
                        f'Include title of the poem at the top.',
            agent=self.agent_poet,
            verbose=True,
            expected_output=f'A structured document with the poem and title of the poem at the top..'
        )

        self.task_generate_image = Task(
            description='Generate 1 image that represents the poem text. '
                        f'Aligning with the {self.poetic_style} theme outlined in the poem. '
                        f'Use the full poem content to generate image. Dont summarize it, ise it as is.',
            agent=self.agent_image_generator,
            expected_output='A digital image file that visually represents the poem.',
            context=[self.task_create_poem]
        )

        self.task_format_content = Task(
            description='Format given poem content as a markdown document, '
                        'Including an image after the title and before the poem. Use <br> as a linebreak',
            agent=self.agent_content_formatter,
            expected_output="""The entire poem content formatted in markdown, With linebreak after each verse. image 
            added after the title""",
            context=[self.task_create_poem, self.task_generate_image],
            verbose=True,
            output_file="poem.md"
        )

        self.task_markdown_to_pdf = Task(
            description='Convert a Markdown file to a PDF document, ensuring the preservation of formatting, '
                        'structure, and embedded images using the mdpdf library.',
            agent=self.agent_markdown_to_pdf_creator,
            expected_output='A PDF file generated from the Markdown input, accurately reflecting the content with '
                            'proper formatting. The PDF should be ready for sharing or printing.'
        )

    def crew(self) -> Crew:
        crew = Crew(
            agents=[self.agent_poet, self.agent_creative_poet, self.agent_image_generator, self.agent_content_formatter,
                    self.agent_markdown_to_pdf_creator],
            tasks=[self.task_create_poem, self.task_generate_image, self.task_format_content,
                   self.task_markdown_to_pdf],
            verbose=True,
            process=Process.sequential
        )
        return crew

    def generate(self):
        self.create_agents()
        self.create_tasks()
        crew = self.crew()
        result = crew.kickoff()
        print(result)


if __name__ == "__main__":
    generator = Classics2PoemGenerator(book="The Karamazov Brothers",
                                       author="Fyodor Dostoevsky",
                                       poetic_style="Alexander Pushkin and Philosophical")
    generator.generate()
