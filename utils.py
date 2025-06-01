import os
import re
from typing import List, Dict, Any


def read_file(path: str) -> str:
    """
    Reads a text file from disk and returns its contents as a single string.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_transcript(path: str) -> str:
    """Reads the entire transcript file into a single string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path: str, content: str) -> None:
    """
    Writes `content` to the file at `path`, overwriting if it already exists.
    """
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def extract_content_list(llm_response: str) -> List[str]:
    """
    Extracts content from triple backticks and splits by semicolons.
    
    Args:
        llm_response (str): The raw LLM response containing triple backticks.
    
    Returns:
        List[str]: List of content items split by semicolons, with whitespace stripped.
    
    Raises:
        ValueError: If triple backtick block not found.
    """
    pattern = r"```(?:[a-zA-Z]*)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, llm_response)
    
    if match:
        content = match.group(1).strip()
        # Split by semicolon and strip whitespace from each item
        return [item.strip() for item in content.split(';') if item.strip()]
    else:
        raise ValueError("Expected content enclosed in triple backticks (```...```) was not found.")


def extract_content(llm_response: str) -> str:
    """
    Args:
        llm_response (str): The raw LLM response containing triple backticks.
    
    Returns:
        str: The text inside the triple backticks, with leading/trailing whitespace removed.
    
    Raises:
        ValueError: If triple backtick block not found.
    """
    pattern = r"```(?:[a-zA-Z]*)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, llm_response)
    
    if match:
        return match.group(1).strip()
    else:
        print(llm_response)
        raise ValueError("Expected content enclosed in triple backticks (```...```) was not found.")