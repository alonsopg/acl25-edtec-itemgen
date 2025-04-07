#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.insert(1, '../')
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def preprocess_text(text):
    """
    Remove newlines, duplicate spaces, trim whitespace, and convert to lower case.
    """
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def main():
    # ---------------------------
    # Load and Preprocess Passages
    # ---------------------------
    passages_df = pd.read_csv("../data/vet_passages_2.csv")
    passages_df['passage'] = passages_df['passage'].apply(preprocess_text)
    passages = passages_df['passage'].tolist()
    print("Number of passages:", len(passages))
    
    # ---------------------------
    # Load the Sentence Transformer Model
    # ---------------------------
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # ---------------------------
    # Compute and Normalize Embeddings
    # ---------------------------
    embeddings = model.encode(passages)
    print("Original embeddings shape:", embeddings.shape)
    
    embeddings = embeddings.astype('float32')
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    print("After normalization, embedding shape:", embeddings.shape)
    
    # ---------------------------
    # Build Faiss Index (Inner Product Index)
    # ---------------------------
    d = embeddings.shape[1]
    index_faiss = faiss.IndexFlatIP(d)
    index_faiss.add(embeddings)
    print("Faiss index built with {} vectors.".format(index_faiss.ntotal))
    
    # ---------------------------
    # Save the Index
    # ---------------------------
    index_path = "faiss.index"  # Save in the demo folder
    faiss.write_index(index_faiss, index_path)
    print("Faiss index saved to", index_path)

if __name__ == '__main__':
    main()
