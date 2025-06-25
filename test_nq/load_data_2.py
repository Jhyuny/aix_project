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
print(corpus[0])

corpus_df = pd.DataFrame({
    "doc_id": corpus["id"],
    "title": corpus["title"],
    "text": corpus["text"]
})

corpus_df.to_json("/mnt/aix7101/jeong/aix_project/nq_rag_corpus.json", orient="records", lines=True)