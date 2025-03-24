from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

# 1. 모델 로드
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. PAWS 데이터셋 로드 (validation 세트 일부만 사용)
dataset = load_dataset("paws", "labeled_final", split="validation[:1000]")

# 3. 문장쌍 추출
sent1 = dataset["sentence1"]
sent2 = dataset["sentence2"]
labels = dataset["label"]

# 4. 문장 임베딩 생성
emb1 = model.encode(sent1)
emb2 = model.encode(sent2)

# 5. 유사도 계산
similarities = cosine_similarity(emb1, emb2)
similarity_scores = similarities.diagonal()  # 문장쌍의 유사도만 추출

# 6. 임계값 기준 이진 분류
threshold = 0.6
preds = (similarity_scores > threshold).astype(int)

# 7. 정확도 계산
acc = accuracy_score(labels, preds)
print(f"Accuracy: {acc:.4f}")