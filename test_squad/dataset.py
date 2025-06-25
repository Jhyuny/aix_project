import os
import pickle
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import json

# Dataset loading and processing
def load_dataset_from_huggingface():
    dataset = load_dataset("nq_open", split="train[:1000]")
    return dataset

# Pickle functions
def save_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

# DPR embeddings generation
def generate_dpr_embeddings(corpus):
    dpr_ctx_encoder = SentenceTransformer('facebook-dpr-ctx_encoder-multiset-base')
    embeddings = dpr_ctx_encoder.encode(corpus, batch_size=64, convert_to_tensor=False, show_progress_bar=True)
    return embeddings

def generate_query_embeddings(queries):
    dpr_question_encoder = SentenceTransformer('facebook-dpr-question_encoder-multiset-base')
    query_embeddings = dpr_question_encoder.encode(queries, batch_size=64, convert_to_tensor=False, show_progress_bar=True)
    return query_embeddings

# FAISS index creation for DPR
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index