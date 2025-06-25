import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

import torch
from transformers import (
    DPRContextEncoder, DPRContextEncoderTokenizer,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer
)

# -----------------------------
# Step 1: 임베딩 생성 (Context, Question)
# -----------------------------
output_dir = "/mnt/aix7101/jeong/aix_project"

corpus_path = os.path.join(output_dir, "nq_rag_corpus.json")
qa_path = os.path.join(output_dir, "nq_rag_qa_pairs.json")

# 데이터 로드
with open(corpus_path, "r", encoding="utf-8") as f:
    corpus_records = json.load(f)
corpus_df = pd.DataFrame(corpus_records)

with open(qa_path, "r", encoding="utf-8") as f:
    qa_records = json.load(f)
qa_pairs_df = pd.DataFrame(qa_records)

# DPR 모델 로드
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")

q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")
q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")

ctx_encoder.eval()
q_encoder.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ctx_encoder.to(device)
q_encoder.to(device)

# Context 임베딩 생성
batch_size = 32
ctx_embeddings = []

for i in tqdm(range(0, len(corpus_df), batch_size), desc="Encoding contexts"):
    batch_texts = corpus_df["text"].iloc[i:i+batch_size].tolist()
    batch_texts = [str(t).strip() for t in batch_texts]

    inputs = ctx_tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = ctx_encoder(**inputs)
        emb_batch = output.pooler_output.cpu().numpy()
        ctx_embeddings.append(emb_batch)

ctx_embeddings = np.vstack(ctx_embeddings)

# Question 임베딩 생성
q_embeddings = []
questions = qa_pairs_df["question"].tolist()

for i in tqdm(range(0, len(questions), batch_size), desc="Encoding questions"):
    batch_questions = questions[i:i+batch_size]
    batch_questions = [str(q).strip() for q in batch_questions]

    inputs = q_tokenizer(batch_questions, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = q_encoder(**inputs)
        emb_batch = output.pooler_output.cpu().numpy()
        q_embeddings.append(emb_batch)

q_embeddings = np.vstack(q_embeddings)

# -----------------------------
# Step 3: 임베딩 저장
# -----------------------------
embedding_dir = "/mnt/aix7101/jeong/aix_project"
os.makedirs(embedding_dir, exist_ok=True)

ctx_path = os.path.join(embedding_dir, "nq_dpr_ctx_embeddings_multiqa.npy")
q_path = os.path.join(embedding_dir, "nq_dpr_q_embeddings_multiqa.npy")

np.save(ctx_path, ctx_embeddings)
np.save(q_path, q_embeddings)

print(f"✅ Context embeddings saved to: {ctx_path}")
print(f"✅ Question embeddings saved to: {q_path}")