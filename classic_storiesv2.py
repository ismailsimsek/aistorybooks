import autogen
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

import chromadb
from tools import *

config_list = [
    {"model": "gpt-3.5-turbo-0125", "api_key": Config.OPENAI_API_KEY},
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
        self.boss = autogen.UserProxyAgent(
            name="Boss",
            system_message="The boss who ask questions and give tasks.",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=6,
            code_execution_config={"use_docker": False},  # we don't want to execute code in this case.
        )

        self.boss_aid = RetrieveUserProxyAgent(
            name="Boss_Assistant",
            default_auto_reply="Reply `TERMINATE` if the task is done.",
            system_message="Assistant who has extra content retrieval power for solving difficult problems.",
            description="Assistant who can retrieve content from documents.",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5,
            retrieve_config={
                "task": "qa",
                "docs_path": self.CURRENT_DIR.joinpath("books/The-Brothers-Karamazov.pdf").as_posix(),
                "chunk_token_size": 2000,
                "client": chromadb.PersistentClient(path=self.CURRENT_DIR.joinpath("chromadb").as_posix()),
                "get_or_create": True,
            },
            code_execution_config={"use_docker": False},  # we don't want to execute code in this case.
        )

        self.author_agent = autogen.AssistantAgent(
            name="Author",
            llm_config=self.llm_config,
            system_message="You are an author writing stories with detailed character descriptions "
                           "and the main plot points. "
            #               "Reply `TERMINATE` in the end when everything is done."
            ,
            description="A creative author specialized in writing stories."
        )

        self.translator = autogen.AssistantAgent(
            name="Translator",
            llm_config=self.llm_config,
            system_message="You are a language translator. "
            #                "Reply `TERMINATE` in the end when everything is done."
            ,
            description="Language translator who can translate between multiple languages."
        )

        self.groupchat = autogen.GroupChat(
            agents=[self.boss, self.boss_aid, self.translator, self.author_agent],
            messages=[],
            max_round=20
        )
        self.manager = autogen.GroupChatManager(groupchat=self.groupchat, llm_config=self.llm_config)

    def generate(self):
        self.boss.initiate_chat(
            recipient=self.manager,
            message=f'First Retrieve and create an summary of "{self.book}" from "{self.author}", detailing a title '
                    f'and character descriptions and the main plot points. '
                    f'Summary size should be minimum {self.summary_size}. '
                    f'Then make the story {self.writing_style} style. '
                    f'And then translate the summary to "{self.language} language" and make it "{self.level}" level.'
        )


if __name__ == "__main__":
    generator = Classics2StoryBookGenerator(book="The Karamazov Brothers",
                                            author="Fyodor Dostoevsky",
                                            language="German",
                                            level="A2 Beginner",
                                            summary_size="10 Chapters, each chapter more than 100 sentences log",
                                            writing_style="Philosophical")
    generator.generate()
