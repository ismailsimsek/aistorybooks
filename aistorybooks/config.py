import os
from pathlib import Path


def _load_api_key(filename: str, env_var: str = None) -> str | None:
    """Loads an API key from an environment variable or a file."""
    if env_var and os.environ.get(env_var):
        return os.environ.get(env_var)

    filepath = Path(__file__).parent.joinpath(filename)
    if filepath.exists():
        return filepath.read_text().strip()
    else:
        print(f"Warning: API key file '{filename}' not found.")
        return None


class Config:
    GROQ_API_BASE_URL = "https://api.groq.com/openai/v1"
    GROQ_MODEL_NAME = "llama3-70b-8192"
    OPENAI_API_BASE_URL = "https://api.openai.com/v1/"
    OPENAI_MODEL_NAME = "gpt-4"
    OPENAI_MODEL_GPT_4O_MINI_NAME = "gpt-4o-mini"
    GEMINI_MODEL_NAME = "gemini-2.0-flash-lite"
    GROQ_API_KEY = _load_api_key("groq-api-key.txt")
    OPENAI_API_KEY = _load_api_key("openai-api-key.txt")
    LOCAL_LLM_API_KEY = _load_api_key("local-api-key.txt")
    GEMINI_API_KEY = _load_api_key("gemini-api-key.txt", env_var="GOOGLE_API_KEY")
