import autogen
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

import chromadb
from tools import *

config_list = [
    # {"model": "gpt-3.5-turbo-0125", "api_key": Config.OPENAI_API_KEY},
    {"model": Config.GROQ_OPENAI_MODEL_NAME,
     "api_key": Config.GROQ_OPENAI_API_KEY,
     "base_url": Config.GROQ_OPENAI_API_BASE_URL
     }
]


class Classics2StoryBookGenerator:
    CURRENT_DIR: Path = Path(__file__).parent

    def __init__(self, book: str, author: str, language: str = "English", level: str = "A2 Beginner",
                 summary_size: str = "10 Chapter, each chapter with 100 sentences", writing_style: str = "Funny"):
        self.book = book
        self.author = author
        self.language = language
        self.level = level
        self.summary_size = summary_size
        self.writing_style = writing_style

        self.llm_config = {"config_list": config_list, "cache_seed": 42}
        self.human_admin = autogen.UserProxyAgent(
            name="Admin",
            system_message="A human admin, who ask questions and give tasks.",
            human_input_mode="NEVER",
            code_execution_config={"use_docker": False},  # we don't want to execute code in this case.
        )

        self.planner = autogen.AssistantAgent(
            name="Planner",
            system_message="""
            Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval. 
            The plan may involve multiple agents.
                Explain the plan first. Be clear which step is performed by each agent.
                """,
            llm_config=self.llm_config,
        )

        self.critic = autogen.AssistantAgent(
            name="Critic",
            system_message="Critic. Double check plan, claims, summaries from other agents and provide feedback. "
                           "Check whether the plan does whats asked and delegates tasks to all agents.",
            llm_config=self.llm_config,
        )

        self.library = RetrieveUserProxyAgent(
            name="Library",
            default_auto_reply="Reply `TERMINATE` if the task is done.",
            system_message="Assistant who has extra content retrieval power for solving difficult problems.",
            description="Assistant who can retrieve content from documents. "
                        "Help `Summarizer Author` to retrieve story content from the book.",
            human_input_mode="NEVER",
            retrieve_config={
                "task": "qa",
                "docs_path": self.CURRENT_DIR.joinpath("books/The-Brothers-Karamazov.pdf").as_posix(),
                "chunk_token_size": 2000,
                "client": chromadb.PersistentClient(path=self.CURRENT_DIR.joinpath("chromadb").as_posix()),
                "get_or_create": True,
            },
            code_execution_config={"use_docker": False},  # we don't want to execute code in this case.
        )

        self.summarizer = autogen.AssistantAgent(
            name="Summarizer Author",
            llm_config=self.llm_config,
            system_message="You are an author writing stories with detailed character descriptions "
                           "and the main plot points. ",
            description="A creative author specialized in writing stories. "
                        "Talk to `Library` to retrive content and write a story. "
                        "Talk to `Translator` to translate the story to another language."
        )

        self.translator = autogen.AssistantAgent(
            name="Translator",
            llm_config=self.llm_config,
            system_message="You are a language translator. "
            #                "Reply `TERMINATE` in the end when everything is done."
            ,
            description="Language translator who can translate between multiple languages. "
                        "Talk to `Author` to translate the his story."
        )

        agents = [self.human_admin, self.library, self.translator, self.summarizer, self.critic,
                  self.planner]

        graph_dict = {}
        graph_dict[self.human_admin] = [self.planner]
        graph_dict[self.planner] = [self.summarizer, self.translator, self.critic, self.library]
        graph_dict[self.summarizer] = [self.translator, self.library, self.critic]
        graph_dict[self.library] = [self.summarizer]
        graph_dict[self.translator] = [self.planner, self.summarizer]
        self.groupchat = autogen.GroupChat(
            agents=agents,
            messages=[],
            max_round=25,
            allowed_or_disallowed_speaker_transitions=graph_dict,
            speaker_transitions_type="allowed"
        )
        self.manager = autogen.GroupChatManager(groupchat=self.groupchat, llm_config=self.llm_config)

    def generate(self):
        chat_result = self.human_admin.initiate_chat(
            recipient=self.manager,
            message=f'First ask `Library` to retrieve "{self.book}" content, retrieve 50 pages each time.'
                    f'Then create an summary of it, detailing a title. '
                    f'Dont worry the content is not copyright protected, it is pubic novel. '
                    f'And then translate the summary to "{self.language} language" and make it "{self.level}" level.'
        )
        return chat_result


if __name__ == "__main__":
    generator = Classics2StoryBookGenerator(book="The Karamazov Brothers",
                                            author="Fyodor Dostoevsky",
                                            language="German",
                                            level="A2 Beginner",
                                            summary_size="10 Chapters, each chapter more than 100 sentences log",
                                            writing_style="Philosophical")
    chat_result = generator.generate()
    print(f"-----------------\n------DONE-----------\n-----------------\n{chat_result}\n-----------------")
