
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import torch

from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from dataset import load_jsonl, load_json, build_faiss_index, generate_dpr_embeddings, generate_query_embeddings
from retrieval import bm25_retrieval_with_generation, dpr_retrieval_with_generation, s_dpr_retrieval, hybrid_retrieval, key_retrieval_with_generation
from key_extract import extract_keyphrases_spacy, extract_keyphrases_keybert



generator_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
generator_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# 데이터 경로 설정
dataset_dir = "/mnt/aix7101/jeong/aix_project"
corpus_path = f"{dataset_dir}/nq_rag_corpus.json"
query_path = f"{dataset_dir}/nq_rag_qa_pairs.json"
corpus_emb_path = f"{dataset_dir}/wiki_dpr_embeddings.npy"
corpus_sentence_emb_path = f"{dataset_dir}/wiki_s_dpr_embeddings.npy"
query_emb_path = f"{dataset_dir}/nq_q_embeddings.npy"

# 데이터 로딩
print("Loading corpus and queries...")
corpus = load_jsonl(corpus_path)
corpus_df = pd.DataFrame(corpus)
queries = load_jsonl(query_path)
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

extract_keyphrases_fn = extract_keyphrases_keybert

# bm25_retrieval_with_generation(queries_df, corpus_df, generator_tokenizer, generator_model, k=5)
dpr_retrieval_with_generation(queries_df, corpus_df, query_embeddings, corpus_embeddings, generator_tokenizer, generator_model, k=5)
# s_dpr_retrieval(queries_df, corpus_df, query_embeddings, corpus_embeddings, k=5)
# hybrid_retrieval(queries_df, corpus_df, query_embeddings, corpus_embeddings, k=5)
key_retrieval_with_generation(ctx_tokenizer, ctx_encoder, queries_df, corpus_df, query_embeddings, corpus_embeddings, extract_keyphrases_fn, enerator_tokenizer, generator_model, k=5)
