import json
import pandas as pd
import logging
from pydantic import BaseModel, create_model, ValidationError
from typing import List, Dict
from openai import OpenAI
import instructor
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Setup colored logging with a custom formatter
class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[0;37m",
        "INFO": "\033[0;36m",
        "WARNING": "\033[0;33m",
        "ERROR": "\033[0;31m",
        "CRITICAL": "\033[1;41m",
    }
    RESET = "\033[0m"
    
    def format(self, record):
        log_fmt = self._fmt
        if record.levelname in self.COLORS:
            log_fmt = self.COLORS[record.levelname] + self._fmt + self.RESET
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Avoid duplicate handlers if logger is already configured
logger = logging.getLogger("AutoNuggetizer")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class NuggetOutput(BaseModel):
    nugget: str
    importance: str  
    source_docid: int

    class Config:
        extra = 'ignore'

class AutoNuggetizer:
    def __init__(self, api_key: str, llm_model: str = "gpt-4o", num_threads: int = 4,
                 nuggets_per_category: int = 15, include_wichtig: bool = True, include_ok: bool = True,
                 include_nicht_wichtig: bool = True, verbose: bool = False):
        """
        Initialize the nuggetizer with an API key, LLM model version, thread count,
        number of nuggets per category, category inclusion flags, and verbose mode.
        
        :param verbose: When True, logs the raw and final JSON response for each API call.
        """
        self.api_key = api_key
        self.llm_model = llm_model
        self.num_threads = num_threads
        self.nuggets_per_category = nuggets_per_category
        self.include_wichtig = include_wichtig
        self.include_ok = include_ok
        self.include_nicht_wichtig = include_nicht_wichtig
        self.verbose = verbose
        self.client = instructor.from_openai(OpenAI(api_key=api_key))
        logger.info(f"Initialized AutoNuggetizer with model {self.llm_model}, {self.num_threads} threads, "
                    f"{self.nuggets_per_category} nuggets per category, "
                    f"Include 'Wichtig': {self.include_wichtig}, 'OK': {self.include_ok}, 'Nicht Wichtig': {self.include_nicht_wichtig}, "
                    f"Verbose mode: {self.verbose}.")

    def extract_balanced_nuggets(self, content: str, passage: str, docid: int) -> List[Dict]:
        """
        Uses the LLM to extract nuggets and returns a list of dictionaries.
        In verbose mode, the raw and final JSON responses are logged in a pretty format.
        """
        logger.debug(f"Extracting nuggets for docid {docid}...")

        # Build list of enabled category keys
        category_keys = []
        if self.include_wichtig:
            category_keys.append("Wichtig")
        if self.include_ok:
            category_keys.append("OK")
        if self.include_nicht_wichtig:
            category_keys.append("Nicht Wichtig")
        if not category_keys:
            raise ValueError("At least one category must be selected for extraction.")

        keys_str = ", ".join(category_keys)


        json_instruction = (
            f"Bitte antworte ausschließlich mit einem JSON-Objekt. Das Objekt soll genau die folgenden Schlüssel enthalten: {keys_str}. "
            f"Jeder Schlüssel muss einer Liste mit exakt {self.nuggets_per_category} Elementen zugeordnet sein. "
            "Jeder Eintrag in der Liste muss ein JSON-Objekt mit den Schlüsseln \"nugget\", \"importance\" und \"source_docid\" sein. "
            "Füge keinen weiteren Text oder zusätzliche Informationen hinzu."
        )
        

        docid_instruction = f"Der Schlüssel 'source_docid' muss immer den Wert {docid} haben."


        system_message = (
            "Sie sind ein Experte für Bildungsinhalte. Ihre Aufgabe ist es, wesentliche Aussagen oder Schlüsselinformationen (\"nuggets\") – "
            "Faktengrundlagen oder Aussagen, die aus dem bereitgestellten Inhalt abgeleitet wurden – zu extrahieren. "
            f"{json_instruction} {docid_instruction}"
        )
        

        user_message = (
            f"Gegebenen Inhalt: {content}\n"
            f"Textauszug: {passage}\n\n"
            f"{json_instruction} {docid_instruction}"
        )


        class Config:
            extra = 'ignore'
        fields = {}
        if self.include_wichtig:
            fields["Wichtig"] = (List[NuggetOutput], ...)
        if self.include_ok:
            fields["OK"] = (List[NuggetOutput], ...)
        if self.include_nicht_wichtig:
            fields["Nicht Wichtig"] = (List[NuggetOutput], ...)
        DynamicNuggetsModel = create_model("DynamicNuggetsModel", __config__=Config, **fields)

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                response_model=DynamicNuggetsModel,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
            )
            response_data = response.dict()
            

            extra_keys = set(response_data.keys()) - set(category_keys)
            if extra_keys:
                logger.debug(f"Received extra keys in response for docid {docid}: {extra_keys}")
                
            if self.verbose:
                pretty_response = json.dumps(response_data, indent=2, ensure_ascii=False)
                logger.info(f"Docid {docid} raw response:\n{pretty_response}")
                
        except ValidationError as ve:
            logger.error(f"Validation error for docid {docid}: {ve}")
            return []
        except Exception as e:
            logger.exception(f"Error processing docid {docid}: {e}")
            return []

        extracted_nuggets = []
        for category in category_keys:
            nuggets_list = response_data.get(category)
            if nuggets_list is None:
                logger.warning(f"Missing category '{category}' in response for docid {docid}.")
                continue
            if len(nuggets_list) < self.nuggets_per_category:
                logger.warning(
                    f"Expected {self.nuggets_per_category} nuggets for category '{category}' but got {len(nuggets_list)}. Using available nuggets."
                )
            elif len(nuggets_list) > self.nuggets_per_category:
                logger.warning(
                    f"Received {len(nuggets_list)} nuggets for category '{category}', trimming to {self.nuggets_per_category}."
                )
                nuggets_list = nuggets_list[:self.nuggets_per_category]
            for nugget in nuggets_list:
                nugget_data = nugget.dict() if hasattr(nugget, "dict") else nugget
                if not nugget_data.get("importance"):
                    nugget_data["importance"] = category

                nugget_data["source_docid"] = docid
                extracted_nuggets.append(nugget_data)

        if self.verbose:
            logger.info(f"Docid {docid} final extracted nuggets: {json.dumps(extracted_nuggets, indent=2, ensure_ascii=False)}")
            
        logger.debug(f"Docid {docid}: Extracted {len(extracted_nuggets)} nuggets from selected categories.")
        return extracted_nuggets

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a DataFrame containing 'docid', 'content', and 'passage' columns,
        extracting nuggets for each row using multithreading.
        """
        logger.info("Starting processing of DataFrame...")
        all_nuggets = []

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_docid = {
                executor.submit(self.extract_balanced_nuggets, row['content'], row['passage'], row['docid']): row['docid']
                for _, row in df.iterrows()
            }
            for future in tqdm(as_completed(future_to_docid), total=len(future_to_docid), desc="Processing documents"):
                docid = future_to_docid[future]
                try:
                    result = future.result()
                    if not result:
                        logger.warning(f"No nuggets extracted for docid {docid}.")
                    all_nuggets.extend(result)
                except Exception as e:
                    logger.exception(f"Error processing docid {docid}: {e}")

        if not all_nuggets:
            logger.warning("No nuggets were extracted from any documents. Returning an empty DataFrame.")
            return pd.DataFrame(columns=["nugget", "importance", "source_docid"])

        nuggets_df = pd.DataFrame(all_nuggets)

        nuggets_df['importance'] = nuggets_df['importance'].astype(str).str.strip().str.lower()


        def map_importance(val):
            norm_val = val.strip().lower()
            vital_keywords = ["hoch", "wichtig", "vital", "high", "wesentlich", "fundamental", "entscheidend"]
            ok_keywords = ["mittel", "ok", "medium"]
            not_vital_keywords = ["niedrig", "nicht wichtig", "not vital", "falsch"]

            for keyword in vital_keywords:
                if keyword in norm_val:
                    return "Vital"
            for keyword in ok_keywords:
                if keyword in norm_val:
                    return "OK"
            for keyword in not_vital_keywords:
                if keyword in norm_val:
                    return "Not Vital"
            return "OK"

        nuggets_df['importance'] = nuggets_df['importance'].apply(map_importance)
        
        logger.info("Completed processing of DataFrame.")
        return nuggets_df
