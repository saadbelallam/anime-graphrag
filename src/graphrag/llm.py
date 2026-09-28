import json
import re

from openai import OpenAI

from graphrag.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


def chat(system: str, user: str, temperature: float = 0.2) -> str:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


def strip_fences(text: str) -> str:
    return re.sub(r"^```[a-z]*|```$", "", text.strip(), flags=re.MULTILINE).strip()


def chat_json(system: str, user: str) -> dict:
    return json.loads(strip_fences(chat(system, user, temperature=0.0)))
