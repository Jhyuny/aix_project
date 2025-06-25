from datasets import load_dataset
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from sklearn.metrics import f1_score
import numpy as np
import faiss
import torch
import pandas as pd
from tqdm import tqdm
 
import json
import os

from datasets import load_dataset

# DPR Wikipedia passages 로드
corpus = load_dataset("wiki_dpr", "psgs_w100.nq.no_index", split="train[:5000000]")

# 샘플 확인


corpus_df = pd.DataFrame({
    "doc_id": corpus["id"],
    "title": corpus["title"],
    "text": corpus["text"]
})

# 3. Load DPR model and tokenizer (use multi-qa)
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
# q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")


ctx_encoder.eval()
# q_encoder.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ctx_encoder.to(device)
# q_encoder.to(device)



batch_size = 64
ctx_embeddings = []

for i in tqdm(range(0, len(corpus_df), batch_size), desc="Encoding contexts"):
    batch_texts = corpus_df["text"].iloc[i:i+batch_size].tolist()
    batch_texts = [str(t).strip() for t in batch_texts]

    inputs = ctx_tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = ctx_encoder(**inputs)
        emb_batch = output.pooler_output.cpu().numpy()  # or output.last_hidden_state[:, 0]
        ctx_embeddings.append(emb_batch)

ctx_embeddings = np.vstack(ctx_embeddings)

# numpy로 저장
np.save("/mnt/aix7101/jeong/aix_project/wiki_dpr_embeddings.npy", ctx_embeddings)