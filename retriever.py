import requests
import json
import pandas as pd
import logging
import datetime

class APIRetriever:
    def __init__(self, api_url, api_key, kpe, llm_version, enable_logging=True):
        self.api_url = api_url
        self.api_key = api_key
        self.kpe = kpe
        self.llm_version = llm_version
        self.enable_logging = enable_logging

        # Setup logger if logging is enabled.
        if self.enable_logging:
            self.logger = logging.getLogger("APIRetrieverLogger")
            self.logger.setLevel(logging.DEBUG)
            # Clear existing handlers to avoid duplicates.
            self.logger.handlers = []
            
            # Create file handler with today's date as filename.
            file_name = datetime.datetime.now().strftime("%Y-%m-%d") + ".log"
            file_handler = logging.FileHandler(file_name)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            # Create console handler for INFO level and above.
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('[API Retriever] %(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        else:
            self.logger = None

    def log(self, message):
        """Logs messages if logging is enabled."""
        if self.enable_logging and self.logger:
            self.logger.info(message)
        else:
            print(f"[API Retriever] {message}")

    def search_query(self, query, k=25):
        payload = json.dumps({
            "query": query,
            "k": k,
            "llm_version": self.llm_version,
            "kpe": self.kpe
        })
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        self.log(f"Sending request for query: {query}")
        self.log(f"Using KPE: {self.kpe}")
        response = requests.post(self.api_url, headers=headers, data=payload)
        
        if response.status_code != 200:
            self.log(f"API request failed with status code {response.status_code}: {response.text}")
            return None
        
        # Log the full JSON response to the file (DEBUG level).
        if self.enable_logging and self.logger:
            self.logger.debug("API response: " + json.dumps(response.json(), indent=2, ensure_ascii=False))
        
        return response.json()

    def search_queries(self, queries_df, k=25):
        results = []
        
        for _, row in queries_df.iterrows():
            query_str, qid = row['queries'], row['qid']
            self.log(f"Processing query: {query_str}")
            response_data = self.search_query(query_str, k)
            
            if response_data and "results" in response_data:
                for entry in response_data["results"]:
                    ag_data = entry.get("augmented_gen", {})
                    ag_response = (f"{ag_data.get('question', '')}\n" +
                                   "\n".join(ag_data.get("answers", []))) if ag_data else ""
                    
                    results.append({
                        "docid": entry.get("docid"),
                        "query": query_str,
                        "qid": qid,
                        "content": entry.get("content"),
                        "passage": entry.get("passage"),
                        "similarity": entry.get("similarity"),
                        "ag_response": ag_response,
                        "correct_answer": ag_data.get("correct_answer", ""),
                        "evaluation": ag_data.get("evaluation", ""),
                        "model_type": entry.get("model_type"),
                        "key_points": entry.get("key_points"),
                        "highlighted": entry.get("highlighted")
                    })
        
        self.log("Search completed.")
        return pd.DataFrame(results)
