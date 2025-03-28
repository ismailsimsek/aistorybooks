from pathlib import Path


class Config:
    GROQ_API_BASE_URL = "https://api.groq.com/openai/v1"
    GROQ_MODEL_NAME = "llama3-70b-8192"
    GROQ_API_KEY = Path(__file__).parent.joinpath("groq-api-key.txt").read_text().strip()
    OPENAI_API_BASE_URL = "https://api.openai.com/v1/"
    OPENAI_MODEL_NAME = "gpt-4"
    OPENAI_API_KEY = Path(__file__).parent.joinpath("openai-api-key.txt").read_text().strip()
