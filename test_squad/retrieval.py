import numpy as np
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import torch

from rank_bm25 import BM25Okapi
from tqdm import tqdm
import pandas as pd
import numpy as np



# BM25 retrieval
def bm25_retrieval(qa_pairs: pd.DataFrame, corpus_df: pd.DataFrame, k: int = 5) -> float:
    tokenized_corpus = [doc.split() for doc in corpus_df["text"]]

    bm25 = BM25Okapi(tokenized_corpus)
    hit_count = 0
    for _, row in tqdm(qa_pairs.iterrows(), total=len(qa_pairs), desc="Evaluating BM25 Recall@K"):
        question = row["question"]
        gt_doc_id = row["doc_id"]

        tokenized_query = question.split()
        scores = bm25.get_scores(tokenized_query)
        topk_indices = np.argsort(scores)[::-1][:k]
        topk_doc_ids = corpus_df.iloc[topk_indices]["doc_id"].tolist()

        if gt_doc_id in topk_doc_ids:
            hit_count += 1
    recall_at_k = hit_count / len(qa_pairs)
    print(f"📌 BM25 Recall@{k}: {recall_at_k:.4f}")
    return recall_at_k


# DPR retrieval
def dpr_retrieval(qa_pairs, corpus_df, q_embeddings, ctx_embeddings, k=5):
    assert len(q_embeddings) == len(qa_pairs), "❗ 질문 임베딩 수와 QA 쌍 수가 일치하지 않습니다."

    hit_count = 0
    idx = 0
    for idx, row in tqdm(qa_pairs, total=len(qa_pairs), desc="Evaluating Recall@K"):
        gt_doc_id = row["doc_id"]
        q_emb = q_embeddings[idx]

        scores = np.dot(ctx_embeddings, q_emb)

        topk_indices = np.argsort(scores)[::-1][:k]
        topk_doc_ids = corpus_df.iloc[topk_indices]["doc_id"].tolist()

        if gt_doc_id in topk_doc_ids:
            hit_count += 1
        idx+=1

    recall_at_k = hit_count / len(qa_pairs)
    print(f"📌 Recall@{k}: {recall_at_k:.4f}")
    return recall_at_k


# Sentence DPR retrieval
def s_dpr_retrieval(qa_pairs, corpus_df, q_embeddings, ctx_embeddings, k=5, aggregation="max"):

    assert len(q_embeddings) == len(qa_pairs), "❗ 질문 임베딩 수와 QA 쌍 수가 일치하지 않습니다."

    hit_count = 0
    for idx, row in tqdm(qa_pairs.iterrows(), total=len(qa_pairs), desc="Evaluating DPR-m Recall@K"):
        gt_doc_id = row["doc_id"]
        q_emb = q_embeddings[idx]

        scores = []
        for doc_sents in ctx_embeddings:
            sent_scores = np.dot(doc_sents, q_emb)
            if aggregation == "max":
                score = np.max(sent_scores)
            elif aggregation == "mean":
                score = np.mean(sent_scores)
            else:
                raise ValueError("aggregation은 'max' 또는 'mean'이어야 합니다.")
            scores.append(score)

        scores = np.array(scores)
        topk_indices = np.argsort(scores)[::-1][:k]
        topk_doc_ids = corpus_df.iloc[topk_indices]["doc_id"].tolist()

        if gt_doc_id in topk_doc_ids:
            hit_count += 1
    recall_at_k = hit_count / len(qa_pairs)
    print(f"📌 sentenceDPR Recall@{k} ({aggregation} aggregation): {recall_at_k:.4f}")
    return recall_at_k


# Hybrid retrieval (BM25 + DPR)
def hybrid_retrieval(qa_pairs, corpus_df, q_embeddings, ctx_embeddings, bm25_top_n=300, k=5):

    tokenized_corpus = [doc.split() for doc in corpus_df["text"]]
    bm25 = BM25Okapi(tokenized_corpus)

    hit_count = 0

    for idx, row in tqdm(qa_pairs.iterrows(), total=len(qa_pairs), desc="Evaluating Hybrid Recall@K"):
        question = row["question"]
        gt_doc_id = row["doc_id"]
        q_emb = q_embeddings[idx]  


        tokenized_query = question.split()
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:bm25_top_n]

        candidate_ctx_embs = ctx_embeddings[bm25_top_indices] 
        dpr_scores = np.dot(candidate_ctx_embs, q_emb)         

        topk_local_indices = np.argsort(dpr_scores)[::-1][:k]
        topk_doc_indices = [bm25_top_indices[i] for i in topk_local_indices]
        topk_doc_ids = corpus_df.iloc[topk_doc_indices]["doc_id"].tolist()

        if gt_doc_id in topk_doc_ids:
            hit_count += 1

    recall_at_k = hit_count / len(qa_pairs)
    print(f"📌 Hybrid Recall@{k} (BM25 top-{bm25_top_n} + DPR top-{k}): {recall_at_k:.4f}")
    return recall_at_k

# Ours retrieval
def key_retrieval(ctx_tokenizer, ctx_encoder, qa_pairs, corpus_df, q_embeddings, ctx_embeddings, extract_keyphrases_fn, top_n_per_keyphrase=50, k=5, device="cuda" if torch.cuda.is_available() else "cpu"):
    hit_count = 0

    for idx, row in tqdm(qa_pairs.iterrows(), total=len(qa_pairs), desc="Custom Retrieval Recall@K"):
        question = row["question"]
        gt_doc_id = row["doc_id"]

        # extract keywords
        keyphrases = extract_keyphrases_fn(question)
        if not keyphrases:
            continue

        inputs = ctx_tokenizer(
            keyphrases,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            phrase_embs = ctx_encoder(**inputs).pooler_output.cpu().numpy()  # shape: (num_phrases, dim)

        # candidates
        candidate_indices = set()
        for emb in phrase_embs:
            scores = np.dot(ctx_embeddings, emb)
            top_indices = np.argsort(scores)[::-1][:top_n_per_keyphrase]
            candidate_indices.update(top_indices)

        if not candidate_indices:
            continue

        query_emb = q_embeddings[idx]

        candidate_indices = list(candidate_indices)
        candidate_embs = ctx_embeddings[candidate_indices]
        rerank_scores = np.dot(candidate_embs, query_emb)

        top_k_indices = np.argsort(rerank_scores)[::-1][:k]
        top_k_doc_ids = corpus_df.iloc[[candidate_indices[i] for i in top_k_indices]]["doc_id"].tolist()

        if gt_doc_id in top_k_doc_ids:
            hit_count += 1

    recall_at_k = hit_count / len(qa_pairs)
    print(f"📌 Custom Retrieval Keyphrase-based Recall@{k}: {recall_at_k:.4f}")
    return recall_at_k