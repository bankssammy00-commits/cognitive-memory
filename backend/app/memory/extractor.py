import os
import json

from dotenv import load_dotenv
from google import genai

from app.models.memory_extraction import ExtractedMemory


load_dotenv()


class MemoryExtractor:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        self.client = genai.Client(api_key=api_key)

    def extract(self, text: str) -> list[ExtractedMemory]:
        prompt = f"""
You are the memory extraction component of a cognitive AI memory system.

Analyze the user's message and identify information that could be
useful to remember for future conversations.

Only extract information that is genuinely worth remembering.

Possible memory types:
- fact
- preference
- event
- decision
- goal
- constraint
- relationship
- plan

Return ONLY valid JSON.

The JSON must be an array.

Each item must contain:

{{
  "content": "a concise description of the memory",
  "memory_type": "one of the allowed types",
  "confidence": 0.0,
  "importance": 0.0,
  "entities": [],
  "topics": [],
  "source_text": "the relevant original text",
  "reasoning": "brief explanation of why this is worth remembering"
}}

Rules:

1. confidence must be between 0 and 1.
2. importance must be between 0 and 1.
3. Do not invent facts.
4. Do not turn temporary conversational filler into memories.
5. Preserve important context.
6. source_text must come directly from the user's message.
7. If there is nothing worth remembering, return [].

USER MESSAGE:

{text}
"""

        response = self.client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
        )

        raw = response.text.strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]

        data = json.loads(raw)

        return [
            ExtractedMemory.model_validate(item)
            for item in data
        ]