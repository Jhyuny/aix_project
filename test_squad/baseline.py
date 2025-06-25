import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import torch

from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

from dataset import load_json, build_faiss_index, generate_dpr_embeddings, generate_query_embeddings
from retrieval import bm25_retrieval, dpr_retrieval, s_dpr_retrieval, hybrid_retrieval, key_retrieval
from key_extract import extract_keyphrases_spacy, extract_keyphrases_keybert



# 데이터 경로 설정
dataset_dir = "/mnt/aix7101/jeong/aix_project"
corpus_path = f"{dataset_dir}/squad_rag_corpus.json"
query_path = f"{dataset_dir}/squad_rag_qa_pairs.json"
corpus_emb_path = f"{dataset_dir}/squad_dpr_ctx_embeddings_multiqa.npy"
corpus_sentence_emb_path = f"{dataset_dir}/squad_dpr_s_ctx_embeddings_multiqa.npy"
query_emb_path = f"{dataset_dir}/squad_dpr_q_embeddings_multiqa.npy"

# 데이터 로딩
print("Loading corpus and queries...")
corpus = load_json(corpus_path)
corpus_df = pd.DataFrame(corpus)
queries = load_json(query_path)
queries_df = pd.DataFrame(queries)

# 임베딩 로딩
print("Loading pre-computed embeddings...")
corpus_embeddings = np.load(corpus_emb_path)
query_embeddings = np.load(query_emb_path)

ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")

# q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")
# q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")

ctx_encoder.eval()
# q_encoder.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ctx_encoder.to(device)
# q_encoder.to(device)

# FAISS 인덱스 빌드
faiss_index = build_faiss_index(corpus_embeddings)

# 평가 함수
def evaluate_retrieval(retrieval_func, qa_pairs, corpus_df, topk=5):
    hit_count = 0
    all_topk_indices = retrieval_func(qa_pairs, corpus_df, topk=topk)
    
    for idx, row in qa_pairs.iterrows():
        gt_doc_id = row["doc_id"]
        topk_indices = all_topk_indices[idx]
        topk_doc_ids = corpus_df.iloc[topk_indices]["doc_id"].tolist()

        if gt_doc_id in topk_doc_ids:
            hit_count += 1

    recall_at_k = hit_count / len(qa_pairs)
    print(f"Recall@{topk}: {recall_at_k:.4f}")
    return recall_at_k

extract_keyphrases_fn = extract_keyphrases_keybert

bm25_retrieval(queries_df, corpus_df, k=5)
dpr_retrieval(queries_df, corpus_df, query_embeddings, corpus_embeddings, k=5)
s_dpr_retrieval(queries_df, corpus_df, query_embeddings, corpus_embeddings, k=5)
hybrid_retrieval(queries_df, corpus_df, query_embeddings, corpus_embeddings, k=5)
key_retrieval(ctx_tokenizer, ctx_encoder, queries_df, corpus_df, query_embeddings, corpus_embeddings, extract_keyphrases_fn, k=5)