#!/usr/bin/env python
# coding: utf-8
import os
import sys
from pathlib import Path

# Determine the project root (one level above demo/)
project_root = Path(__file__).resolve().parent.parent
print("Project root:", project_root)
sys.path.insert(0, str(project_root))

from KeyPointExtraction import KPExtractor
from AugmentedGeneration import generate_augmented_question

# Build absolute path for the CSV file
data_file = project_root / "data" / "vet_passages_2.csv"

import re
import hashlib
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import concurrent.futures
from flask_cors import CORS
from functools import wraps

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type"])

# ---------------------------
# Colored Debug Message Function
# ---------------------------
def debug_msg(message, level="DEBUG"):
    colors = {
        "DEBUG": "\033[94m",   # Blue
        "INFO": "\033[92m",    # Green
        "WARNING": "\033[93m", # Yellow
        "ERROR": "\033[91m",   # Red
        "RESET": "\033[0m"
    }
    print(f"{colors.get(level, colors['DEBUG'])}[{level}] {message}{colors['RESET']}")

# ---------------------------
# API Key Authentication Setup using Hashes
# ---------------------------
VALID_API_KEY_HASHES = {
    hashlib.sha256("l93048JSDOLPJHkjhku59C".encode('utf-8')).hexdigest()
}

def require_api_key(view_function):
    @wraps(view_function)
    def decorated_function(*args, **kwargs):
        # Hardcode the API key internally; ignore incoming headers.
        api_key = "l93048JSDOLPJHkjhku59C"
        hashed_key = hashlib.sha256(api_key.encode('utf-8')).hexdigest()
        if hashed_key not in VALID_API_KEY_HASHES:
            debug_msg("Hardcoded API key is invalid.", "WARNING")
            return jsonify({"error": "Invalid API key"}), 401
        return view_function(*args, **kwargs)
    return decorated_function

# ---------------------------
# Preprocessing Function
# ---------------------------
def preprocess_text(text):
    """
    Entfernt Zeilenumbrüche, doppelte Leerzeichen, trimmt den Text und wandelt ihn in Kleinbuchstaben um.
    """
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

# ---------------------------
# Load Passages Data
# ---------------------------
debug_msg(f"Loading passages data from '{data_file}'", "INFO")
passages_df = pd.read_csv(data_file)
# Preprocess only the passage column used for retrieval.
passages_df['passage'] = passages_df['passage'].apply(preprocess_text)
debug_msg(f"Number of passages: {len(passages_df)}", "DEBUG")

# ---------------------------
# Load Sentence Transformer Model for Query Encoding
# ---------------------------
debug_msg("Loading Sentence Transformer model for query encoding...", "INFO")
query_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
debug_msg("Query encoding model loaded.", "DEBUG")

# ---------------------------
# Load the Precomputed Faiss Index
# ---------------------------
index_path = "faiss.index"  # Ensure the correct path to your Faiss index
debug_msg(f"Loading precomputed Faiss index from {index_path}", "INFO")
index_faiss = faiss.read_index(index_path)
debug_msg(f"Loaded Faiss index with {index_faiss.ntotal} vectors.", "DEBUG")

# ---------------------------
# Define Query Function
# ---------------------------
def get_results_df(query_str, k=10):
    debug_msg("Preprocessing and encoding query.", "INFO")
    query_str = preprocess_text(query_str)
    q_emb = query_model.encode(query_str).astype('float32')
    # Normalize the embedding
    q_emb = q_emb / np.linalg.norm(q_emb)
    q_emb = np.expand_dims(q_emb, axis=0)

    debug_msg("Performing Faiss search.", "INFO")
    distances, indices = index_faiss.search(q_emb, k)
    results_df = passages_df.iloc[indices[0]]
    return results_df, indices[0], distances[0]

# ---------------------------
# Initialize KPExtractor (using the FLAN-T5 based implementation)
# ---------------------------
debug_msg("Initializing KPExtractor...", "INFO")
kp_extractor = KPExtractor(
    similarity_threshold=0.75, 
    lambda_param=0.8
)
debug_msg("KPExtractor initialized.", "DEBUG")

# ---------------------------
# Helper Function for Multi-threading in KeyPoint Extraction
# ---------------------------
def process_passage(passage):
    debug_msg("Starting keypoint extraction for a passage.", "INFO")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as inner_executor:
        future_kp = inner_executor.submit(kp_extractor.get_key_points, [passage], 7, True)
        kp_out = future_kp.result()
    debug_msg("Finished keypoint extraction for a passage.", "DEBUG")
    return kp_out

# ---------------------------
# Define Flask API Routes with Authentication
# ---------------------------
@app.route('/search', methods=['POST'])
@require_api_key
def search():
    debug_msg("Received search request.", "INFO")
    data = request.get_json()
    if not data or 'query' not in data:
        debug_msg("No query provided in the request.", "WARNING")
        return jsonify({"error": "Please provide a 'query' in the request body."}), 400

    query_str = data['query']
    k = data.get('k', 10)
    llm_version = data.get("llm_version", "GPT")
    # Validate llm_version (accepted: GPT, GPT-4O-MINI, DEEPSEEK)
    valid_versions = {"GPT", "GPT-4O", "DEEPSEEK"}
    if llm_version.upper() not in valid_versions:
        return jsonify({"error": "Invalid llm_version provided. Please choose 'GPT' (or 'gpt-4o-mini') or 'DEEPSEEK'."}), 400

    kpe_mode = str(data.get("kpe", "True")).lower() == "true"

    try:
        results_df, indices, distances = get_results_df(query_str, k=k)
    except Exception as e:
        debug_msg(f"Error during search: {str(e)}", "ERROR")
        return jsonify({"error": str(e)}), 500

    debug_msg("Processing retrieval results.", "INFO")
    final_results = []
    for row, sim in zip(results_df.itertuples(), distances):
        docid = int(row.docid) if hasattr(row, 'docid') else None

        if kpe_mode:
            # Key-Point Extraction mode.
            try:
                kp_res = process_passage(row.passage)
                gen_input = kp_res["key_points"]
                returned_key_points = kp_res["key_points"]
                returned_highlighted = kp_res["highlighted_passages"]
            except Exception as e:
                debug_msg(f"Error during key point extraction: {str(e)}", "ERROR")
                gen_input = []
                returned_key_points = ["None"]
                returned_highlighted = ["None"]
        else:
            # No key-point extraction mode: use full passage text from 'content'
            gen_input = row.content
            returned_key_points = ["None"]
            returned_highlighted = ["None"]

        try:
            augmented_gen = generate_augmented_question(gen_input, llm_version, use_kpe=kpe_mode)
        except Exception as e:
            debug_msg(f"Error during augmented generation: {str(e)}", "ERROR")
            augmented_gen = {
                "question": "Error generating question.",
                "answers": [],
                "correct_answer": "",
                "evaluation": ""
            }

        final_results.append({
            "model_type": llm_version,
            "docid": docid,
            "passage": row.passage,
            "content": row.content,
            "similarity": float(sim),
            "augmented_gen": augmented_gen,
            "query": query_str,
            "kpe": "True" if kpe_mode else "False",
            "key_points": returned_key_points,
            "highlighted": returned_highlighted
        })

    debug_msg("Search request processed successfully.", "INFO")
    return jsonify({"results": final_results})

@app.route('/test', methods=['GET'])
@require_api_key
def test():
    debug_msg("Received test request.", "INFO")
    return jsonify({"message": "API is working"})

# ---------------------------
# Run the App
# ---------------------------
if __name__ == '__main__':
    debug_msg("Starting Flask app...", "INFO")
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
