import os
from openai import OpenAI
import instructor
from pydantic import BaseModel
from typing import Optional

class MCQuestion(BaseModel):
    question: str
    answers: list[str]
    correct_answer: str

def generate_augmented_question(input_text, llm_version="gpt-4o", use_kpe=True):
    """
    Creates a multiple choice question in German based on either
    the extracted key points (if use_kpe=True) or the full text
    (if use_kpe=False).
    
    The output is a JSON object with the following keys:
      - "question": The generated question text.
      - "answers": A list of four answer options.
      - "correct_answer": The correct answer option.
    
    :param input_text: If use_kpe=True, a list of key points; if False, a string with the full text.
    :param llm_version: 'gpt-4o-mini' (or 'GPT') or 'DEEPSEEK'.
    :param use_kpe: Boolean whether to use key point extraction.
    :return: A dictionary with the generated question and answer data.
    """
    if use_kpe:
        prompt = (
            "Ihre Aufgabe ist es, eine Multiple-Choice-Frage zu erstellen, die auf den Top 15 Schlüsselaussagen " +
            "\n".join(f"- {kp}" for kp in input_text) +
            " basiert. Die Ausgabe sollte ein JSON-Objekt im Format sein: " +
            "{\"question\": \"<Question text>\", \"answers\": [\"Antwort 1\", \"Antwort 2\", \"Antwort 3\", \"Antwort 4\"], \"correct_answer\": \"<richtige Antwort>\"}. " +
            "Die Frage muss so formuliert sein, dass sie nur mit den angegebenen Antwortmöglichkeiten beantwortet werden kann und nur eine richtige Antwort existiert. " +
            "Speichern Sie die richtige Antwort im JSON-Feld \"correct_answer\"."
        )
    else:
        prompt = (
            "Ihre Aufgabe ist es, eine Multiple-Choice-Frage zu erstellen, die auf den 15 wichtigsten Aussagen des folgenden Textes " +
            input_text +
            " basiert. Die Ausgabe sollte ein JSON-Objekt im Format sein: " +
            "{\"question\": \"<Question text>\", \"answers\": [\"Antwort 1\", \"Antwort 2\", \"Antwort 3\", \"Antwort 4\"], \"correct_answer\": \"<richtige Antwort>\"}. " +
            "Die Frage muss so formuliert sein, dass sie ausschließlich anhand der verfügbaren Antwortmöglichkeiten beantwortet werden kann und nur eine richtige Antwort besitzt. " +
            "Speichern Sie die richtige Antwort im JSON-Feld \"correct_answer\"."
        )

    model_tag = llm_version.upper()
    if model_tag in ["GPT", "GPT-4O"]:
        client = instructor.from_openai(OpenAI(api_key=""))
        responses = client.chat.completions.create_iterable(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_model=MCQuestion,
            temperature=0
        )
        mc_question = next(responses)
        mc_question_dict = mc_question.dict()
        output = mc_question_dict
    elif model_tag == "DEEPSEEK":
        client = instructor.from_openai(
            OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        )
        mc_question = client.chat.completions.create(
            model="DEEPSEEK",
            messages=[{"role": "user", "content": prompt}],
            response_model=MCQuestion,
            temperature=0
        )
        mc_question_dict = mc_question.dict()
        output = mc_question_dict
    else:
        raise ValueError("Invalid llm_version.")

    return output
