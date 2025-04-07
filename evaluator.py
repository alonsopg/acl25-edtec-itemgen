import os
import time
import pandas as pd
import numpy as np
import json
import re
from openai import OpenAI
import instructor
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pydantic import BaseModel, ValidationError


class LLMBasedResponse(BaseModel):
    Vstrict_LLM: float
    V_LLM: float
    W_LLM: float
    A_LLM: float


class ClarityResult(BaseModel):
    grammar_score: int       
    succinctness_score: int  
    readability_score: int   
    explanation: str        

def extract_json(text: str) -> str:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return ""

class Evaluator:
    def __init__(
        self,
        nuggets_file,
        eval_llm_based=True,
        eval_clarity=True,
        eval_length=True,
        num_threads=11,
        seed=42,
        enable_logging=False, 
        llm_version='gpt-4o'
    ):
        
        # Load nuggets
        self.nuggets_df = pd.read_csv(nuggets_file)
        
        # Set evaluation method flags
        self.eval_llm_based = eval_llm_based
        self.eval_clarity = eval_clarity
        self.eval_length = eval_length
        
        # Threading and reproducibility
        self.num_threads = num_threads
        self.seed = seed
        np.random.seed(seed)
        

        self.enable_logging = enable_logging
        

        self.llm_version = llm_version
  
        self.openai_api_key =
        

        self.clarity_client = instructor.from_openai(OpenAI(api_key=self.openai_api_key))
        

        self.weights = {'Vital': 1, 'OK': 0.5, 'Not Vital': 0}
    
    def log(self, message, color='green'):
        if self.enable_logging:
            colors = {
                'red': '\033[91m',
                'green': '\033[92m',
                'yellow': '\033[93m',
                'blue': '\033[94m',
                'magenta': '\033[95m',
                'cyan': '\033[96m',
                'reset': '\033[0m'
            }
            print(f"{colors.get(color, colors['green'])}{message}{colors['reset']}")
    
    def calculate_clarity(self, question):
        if not self.eval_clarity:
            return None, None, None, None
        start = time.time()
        try:
            self.log(f"Starting clarity evaluation for question (len={len(question.split())} words)", color="cyan")
            
            # System prompt from Table row 1 (German version)
            system_prompt = (
                "Du bist ein Experte für die Bewertung der Qualität von Prüfungsfragen. Gib ausschließlich ein JSON-Objekt mit den Schlüßeln: "
                "\"grammar_score\" (Ganzzahl 1 bis 5), \"succinctness_score\" (Ganzzahl 1 bis 5), \"readability_score\" (Ganzzahl 1 bis 5), \"explanation\" (kurze Begründung). "
                "Beispielformat: {\"grammar_score\": 4, \"succinctness_score\": 3, \"readability_score\": 5, \"explanation\": \"Die Frage ist grammatisch gut, sehr klar...\"}. "
                "Kein zusätzlicher Text oder andere Schlüßel."
            )
            
            # User prompt from Table row 2 (German version)
            user_prompt = (
                f"Bewerten Sie die folgende Multiple-Choice-Prüfungsfrage in deutscher Sprache: {question}. "
                "Bewerten Sie sie auf einer Skala von 1,0 bis 5,0 für die folgenden Kriterien: "
                "-- Grammatik (grammar_score): Bewerten Sie die grammatikalische Richtigkeit. "
                "-- Prägnanz (succinctness_score): Beurteilen Sie, wie prägnant und direkt die Frage ist. "
                "-- Lesbarkeit (readability_score): Beurteilen Sie, wie leicht die Frage zu verstehen ist. "
                "Die Ergebnisse werden im folgenden JSON-Format zurückgegeben: "
                "{\"grammar_score\": <score>, \"succinctness_score\": <score>, \"readability_score\": <score>, \"explanation\": <Kurzerläuterung zu den angegebenen Punktzahlen>}."
            )
            
            response = self.clarity_client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                response_model=ClarityResult,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            grammar_score = response.grammar_score
            succinctness_score = response.succinctness_score
            readability_score = response.readability_score
            explanation = response.explanation
            
            elapsed = time.time() - start
            self.log(f"Clarity evaluation completed in {elapsed:.2f} sec", color="magenta")
            return grammar_score, succinctness_score, readability_score, explanation
        
        except Exception as e:
            self.log(f"Error in clarity evaluation: {e}", color="red")
            return 0, 0, 0, "Error"
    
    def calculate_nugget_metrics_llm_based(self, candidate_answer, nuggets_df):
        start = time.time()
        nuggets_list = []
        for _, row in nuggets_df.iterrows():
            nuggets_list.append(f"Nugget: {row['nugget']}, Wichtigkeit: {row['importance']}")
        nuggets_text = "\n".join(nuggets_list)
        
        # Updated system prompt with explicit normalization instructions.
        system_prompt = (
            "Sie sind ein Experte für die Bewertung der Qualität von Multiple-Choice-Prüfungsfragen basierend auf vorab bewerteten, "
            "gekennzeichneten Informationsnuggets – relevante Fakten, die die Frage und ihre Antworten abdecken müssen. "
            "Bitte weisen Sie jedem Nugget einen Wert zu: vollständige Unterstützung (support) = 1, teilweise Unterstützung (partial_support) = 0,5, keine Unterstützung (not_support) = 0. "
            "Berechnen Sie anschließend folgende Metriken (bitte geben Sie alle Werte als Zahlen zwischen 0 und 1 an):\n"
            "- A (All) Score: Durchschnitt aller Nugget-Werte.\n"
            "- V (Vital) Score: Durchschnitt der Werte für Nuggets, die als 'Vital' gekennzeichnet sind.\n"
            "- W (Gewichteter) Score: Gewichteter Durchschnitt, wobei 'Vital'-Nuggets mit 1 und 'OK'-Nuggets mit 0,5 gewichtet werden.\n"
            "- Vstrict (Vital Strict) Score: Durchschnitt für 'Vital'-Nuggets, wobei nur volle Unterstützung (support = 1) gezählt wird (partial_support = 0).\n"
            "Geben Sie Ihre Antwort als gültiges JSON-Objekt mit den Schlüsseln: "
            "{\"Vstrict_LLM\": <Score>, \"V_LLM\": <Score>, \"W_LLM\": <Score>, \"A_LLM\": <Score>}."
        )
        
        user_prompt = (
            f"Bewerten Sie die folgende Kombination aus Frage und Kandidatenantwort: {candidate_answer}. "
            f"Liste der Nuggets (Format: Nugget: <Text>, Wichtigkeit: <Vital/OK/Not Vital>):\n{nuggets_text}\n"
            "Für jedes Nugget ist zu bewerten, wie gut es durch die kombinierte Fragestellung und Antwort abgedeckt wird. "
            "Antworten Sie mit einem JSON-Objekt, das die aggregierten Metriken enthält."
        )
        
        try:
            response = self.clarity_client.chat.completions.create(
                model=self.llm_version,
                temperature=0,
                response_model=LLMBasedResponse,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            metrics = response
        except Exception as e:
            self.log(f"Error in LLMBased metrics evaluation: {e}", color="red")
            raw_response = self.clarity_client.chat.completions.create(
                model=self.llm_version,
                temperature=0,
                response_model=str,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            json_str = extract_json(raw_response)
            try:
                metrics = LLMBasedResponse.parse_raw(json_str)
            except ValidationError as e2:
                self.log(f"Fallback parsing failed: {e2}", color="red")
                return (0, 0, 0, 0)
        elapsed = time.time() - start
        self.log(f"LLMBased metrics computed in {elapsed:.2f} sec", color="yellow")
        

        vstrict = max(0, min(metrics.Vstrict_LLM, 1))
        v_val   = max(0, min(metrics.V_LLM, 1))
        w_val   = max(0, min(metrics.W_LLM, 1))
        a_val   = max(0, min(metrics.A_LLM, 1))
        return (vstrict, v_val, w_val, a_val)
    
    def calculate_length(self, candidate_answer):

        if not self.eval_length:
            return None
        start = time.time()
        length = len(candidate_answer.split())
        elapsed = time.time() - start
        self.log(f"Length metric computed in {elapsed:.2f} sec", color="cyan")
        return length
    
    def evaluate_passages_and_questions(self, passages_df, questions_df):

        docids = passages_df['docid'].unique()
        results = []
        overall_start = time.time()
        
        def evaluate_docid(docid):
            self.log(f"Starting evaluation for docid: {docid}", color="green")
            passage = passages_df.query(f"docid == {docid}")['passage'].iloc[0]
            query = questions_df.query(f"docid == {docid}")['query'].iloc[0]
            question = questions_df.query(f"docid == {docid}")['ag_response'].iloc[0]

            nuggets = self.nuggets_df[self.nuggets_df['docid'] == docid]
            
            if pd.isna(question) or nuggets.empty:
                self.log(f"Skipping docid: {docid} aufgrund fehlender Frage oder Nuggets", color="red")
                return None
            
            result = {
                "docid": docid,
                "query": query,
                "passage": passage,
                "ag_response": question
            }
            
            if self.eval_llm_based:
                start_time = time.time()
                vstrict, v_val, w_val, a_val = self.calculate_nugget_metrics_llm_based(question, nuggets)
                result.update({
                    "Vstrict_LLM": vstrict,
                    "V_LLM": v_val,
                    "W_LLM": w_val,
                    "A_LLM": a_val,
                })
                self.log(
                    f"Docid {docid}: LLMBased metrics computed in {(time.time()-start_time):.2f} sec",
                    color="yellow"
                )
            
            if self.eval_clarity:
                start_time = time.time()
                grammar_score, succinctness_score, readability_score, explanation = self.calculate_clarity(question)
                result["grammar_score"] = grammar_score
                result["succinctness_score"] = succinctness_score
                result["readability_score"] = readability_score
                result["clarity_explanation"] = explanation
                self.log(
                    f"Docid {docid}: Clarity evaluation completed in {(time.time()-start_time):.2f} sec",
                    color="magenta"
                )
            
            if self.eval_length:
                start_time = time.time()
                result["L"] = self.calculate_length(question)
                self.log(
                    f"Docid {docid}: Length metric computed in {(time.time()-start_time):.2f} sec",
                    color="cyan"
                )
            
            self.log(f"Finished evaluation for docid: {docid}", color="green")
            return result
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_docid = {executor.submit(evaluate_docid, docid): docid for docid in docids}
            with tqdm(total=len(future_to_docid), desc="Evaluating docs") as pbar:
                for future in as_completed(future_to_docid):
                    docid = future_to_docid[future]
                    res = future.result()
                    if res is not None:
                        results.append(res)
                    elapsed_minutes = (time.time() - overall_start) / 60.0
                    pbar.set_postfix({"Last docid": docid, "Elapsed (min)": f"{elapsed_minutes:.2f}"})
                    pbar.update(1)
                    
        return pd.DataFrame(results).sort_values(by='docid').reset_index(drop=True)
