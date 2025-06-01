import json
import os
from typing import List, Dict, Any
from utils import read_file
from prompts import PROCESS_TEXT_SYS_PROMPT
from openai import OpenAI


# os.environ['OPENAI_API_KEY'] = read_file("openai_key.txt").strip()


def clean_with_llm(raw_transcript: str, model: str = "gpt-4o", temperature: float = 0.0) -> Dict[str, Any]:
    """
    Sends `raw_transcript` to GPT-4 (or specified model) with the PROCESS_TEXT_SYS_PROMPT instructions,
    and receives back a JSON object conforming to the { "utterances": [ ... ] } schema.
    """
    client = OpenAI()

    messages = [
        {"role": "system", "content": PROCESS_TEXT_SYS_PROMPT},
        {"role": "user", "content": raw_transcript}
    ]

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
            response_format={"type": "json_object"}
        )
        content = resp.choices[0].message.content

        if not content:
            raise ValueError("Empty response content from LLM.")

        parsed = json.loads(content)
        return parsed

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {content[:300]}") from e

    except Exception as e:
        raise RuntimeError(f"Failed to get LLM response: {e}") from e


def process_and_save(input_txt_path: str, output_json_path: str) -> None:
    """
    Reads a raw transcript from `input_txt_path`, calls `clean_with_llm(...)`, 
    and writes the resulting JSON to `output_json_path`.
    """
    with open(input_txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    result_json = clean_with_llm(raw_text)

    if "utterances" not in result_json or not isinstance(result_json["utterances"], list):
        raise ValueError("LLM did not return an object with key 'utterances' as a list.")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    print(f"Structured utterances written to: {output_json_path}")
