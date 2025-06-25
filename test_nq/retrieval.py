import numpy as np
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import torch

from rank_bm25 import BM25Okapi
from tqdm import tqdm
import pandas as pd
import numpy as np


import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


# 간단한 토큰 기반 F1 계산기
def compute_simple_f1(references, predictions):
    f1_scores = []
    for ref, pred in zip(references, predictions):
        ref_tokens = set(ref.lower().split())
        pred_tokens = set(pred.lower().split())
        common = ref_tokens & pred_tokens
        if len(common) == 0:
            f1_scores.append(0)
            continue
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)
    return np.mean(f1_scores)


# 기존 BM25 + generator 평가용 함수
def bm25_retrieval_with_generation(
    qa_pairs: pd.DataFrame,
    corpus_df: pd.DataFrame,
    generator_tokenizer,
    generator_model,
    k: int = 5,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    generator_model.to(device)
    generator_model.eval()

    tokenized_corpus = [doc.split() for doc in corpus_df["text"]]
    bm25 = BM25Okapi(tokenized_corpus)

    predictions = []
    references = []

    for _, row in tqdm(qa_pairs.iterrows(), total=len(qa_pairs), desc="BM25 + Generator Evaluation"):
        question = row["question"]
        gt_answer = row["answer"]

        tokenized_query = question.split()
        scores = bm25.get_scores(tokenized_query)
        topk_indices = np.argsort(scores)[::-1][:k]
        topk_passages = corpus_df.iloc[topk_indices]["text"].tolist()

        # generator에 넣을 input 생성
        prompt = f"question: {question} context: {' '.join(topk_passages)}"
        inputs = generator_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = generator_model.generate(**inputs, max_length=64)
            answer = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

        predictions.append(answer.strip())
        references.append(gt_answer.strip())

    acc = accuracy_score(references, predictions)
    f1 = compute_simple_f1(references, predictions)

    print(f"📌 BM25+Generator Accuracy: {acc:.4f}")
    print(f"📌 BM25+Generator F1 Score: {f1:.4f}")

    return acc, f1


# DPR retrieval
def dpr_retrieval_with_generation(
    qa_pairs, 
    corpus_df, 
    q_embeddings, 
    ctx_embeddings, 
    generator_tokenizer, 
    generator_model, 
    k=5, 
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    generator_model.to(device)
    generator_model.eval()

    predictions = []
    references = []

    for idx, row in tqdm(qa_pairs.iterrows(), total=len(qa_pairs), desc="DPR + Generator Evaluation"):
        question = row["question"]
        gt_answer = row["answer"]
        q_emb = q_embeddings[idx]

        # DPR retrieval
        scores = np.dot(ctx_embeddings, q_emb)
        topk_indices = np.argsort(scores)[::-1][:k]
        topk_passages = corpus_df.iloc[topk_indices]["text"].tolist()

        # generator input 생성
        prompt = f"question: {question} context: {' '.join(topk_passages)}"
        inputs = generator_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = generator_model.generate(**inputs, max_length=64)
            answer = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

        predictions.append(answer.strip())
        references.append(gt_answer.strip())

    acc = accuracy_score(references, predictions)
    f1 = compute_simple_f1(references, predictions)

    print(f"📌 DPR+Generator Accuracy: {acc:.4f}")
    print(f"📌 DPR+Generator F1 Score: {f1:.4f}")

    return acc, f1


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
def key_retrieval_with_generation(
    ctx_tokenizer,
    ctx_encoder,
    qa_pairs,
    corpus_df,
    q_embeddings,
    ctx_embeddings,
    extract_keyphrases_fn,
    generator_tokenizer,
    generator_model,
    top_n_per_keyphrase=300,
    k=5,
):
    # DataParallel 적용 (모든 GPU 사용)
    generator_model = torch.nn.DataParallel(generator_model)
    generator_model.to("cuda")
    device = next(generator_model.parameters()).device
    generator_model.eval()

    predictions = []
    references = []

    for idx, row in tqdm(qa_pairs.iterrows(), total=len(qa_pairs), desc="Keyphrase + Generator Evaluation"):
        question = row["question"]
        gt_answer = row["answer"]

        # 1️⃣ Keyphrase 추출
        keyphrases = extract_keyphrases_fn(question)
        if not keyphrases:
            continue

        inputs = ctx_tokenizer(
            keyphrases,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            phrase_embs = ctx_encoder(**inputs).pooler_output.cpu().numpy()

        # 2️⃣ Candidate documents 선정
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
        top_k_passages = corpus_df.iloc[[candidate_indices[i] for i in top_k_indices]]["text"].tolist()

        # 3️⃣ Generator에 넣기 (디바이스 올리는 부분 수정)
        prompt = f"question: {question} context: {' '.join(top_k_passages)}"
        gen_inputs = generator_tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        gen_inputs = {k: v.to(device) for k, v in gen_inputs.items()}

        with torch.no_grad():
            outputs = generator_model.module.generate(**gen_inputs, max_length=64)
            answer = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

        predictions.append(answer.strip())
        references.append(gt_answer.strip())

    acc = accuracy_score(references, predictions)
    f1 = compute_simple_f1(references, predictions)

    print(f"📌 Keyphrase+Generator Accuracy: {acc:.4f}")
    print(f"📌 Keyphrase+Generator F1 Score: {f1:.4f}")

    return acc, f1