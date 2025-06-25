import spacy
from keybert import KeyBERT
from typing import List
import re



# Load spacy model for rule-based extraction
nlp = spacy.load("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")
WH_WORDS = {"what", "who", "whom", "where", "when", "why", "how"}

def extract_keyphrases_spacy(question: str):
    doc = nlp(question.lower())
    keyphrases = set()

    wh_word = None
    for token in doc:
        if token.text in WH_WORDS:
            wh_word = token.text
            break

    for chunk in doc.noun_chunks:
        if any(not token.is_stop and token.pos_ in {"NOUN", "PROPN"} for token in chunk):
            keyphrases.add(chunk.text.strip())
    if wh_word:
        hint_map = {
            "who": "person",
            "where": "location",
            "when": "time",
            "why": "reason",
            "how": "method",
        }
        hint = hint_map.get(wh_word)
        if hint:
            keyphrases.add(hint)

    return list(keyphrases)



# KeyBERT based extraction
kw_model = KeyBERT(model='all-MiniLM-L6-v2')

def extract_keyphrases_keybert(question: str, top_n: int = 5, diversity: bool = False) -> List[str]:
    question_clean = re.sub(r"[^\w\s]", "", question.lower())  # 간단한 전처리

    if diversity:
        keyphrases = kw_model.extract_keywords(
            question_clean,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            use_mmr=True,
            diversity=0.7,
            top_n=top_n
        )
    else:
        keyphrases = kw_model.extract_keywords(
            question_clean,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            top_n=top_n
        )
    return [phrase for phrase, _ in keyphrases]


# # Unified extraction interface
# def extract_keywords(text, method="spacy", topk=5):
#     if method == "spacy":
#         return spacy_extract(text, topk)
#     elif method == "keybert":
#         return keybert_extract(text, topk)
#     else:
#         raise ValueError("Unknown extraction method")
