import os
import joblib
import pandas as pd
import numpy as np
import requests
from typing import List, Dict, Any

MODEL_PATH = "/mnt/data/models/career_model.joblib"

# Try to load model artifact
_artifact = None
_model = None
_resource_map = {}
_feature_columns = None

try:
    _artifact = joblib.load(MODEL_PATH)
    _model = _artifact.get("model")
    _resource_map = _artifact.get("resource_map", {})
    # infer feature columns from the pipeline if possible (scaler step uses columns from input at predict time)
    # We'll accept that Streamlit/predict callers will provide dicts matching training columns.
    print("Loaded model from", MODEL_PATH)
except Exception as e:
    print("Warning: Could not load model artifact at %s. Error: %s" % (MODEL_PATH, e))
    _model = None
    _resource_map = {}

def predict_from_dict(payload: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Accepts a dict mapping feature_name -> value (numbers). Returns top_k career predictions with probabilities.
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Run train.py and ensure model artifact exists at %s" % MODEL_PATH)
    # Ensure consistent order: use columns from model if available, else sort keys
    if hasattr(_model, "named_steps") and "scaler" in _model.named_steps:
        # can't reliably extract training columns from pipeline; assume caller provides correct keys
        pass
    # Convert to dataframe
    df = pd.DataFrame([payload])
    probs = _model.predict_proba(df)[0]
    classes = _model.classes_
    pairs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"career": c, "prob": float(p)} for c, p in pairs]

def get_resources_for_careers(careers: List[str], top_n_per_career: int = 5) -> Dict[str, List[Dict[str,str]]]:
    """
    Return recommended courses/certifications for given careers from the curated resource map.
    """
    out = {}
    for c in careers:
        out[c] = _resource_map.get(c, [])[:top_n_per_career]
    return out

def get_jobs_from_remotive(career: str, limit:int=10) -> List[Dict[str, str]]:
    """
    Use Remotive public API to fetch remote jobs matching the career keyword.
    (No API key required. Results limited to remote jobs.)
    """
    try:
        url = "https://remotive.io/api/remote-jobs"
        params = {"search": career}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        jobs = []
        for job in data.get("jobs", [])[:limit]:
            jobs.append({
                "title": job.get("title"),
                "company": job.get("company_name"),
                "url": job.get("url"),
                "location": job.get("candidate_required_location"),
                "type": job.get("job_type")
            })
        return jobs
    except Exception as e:
        return [{"error": "Failed to fetch jobs: %s" % e}]

# Simple FAQ and chat helper (rule-based + optional semantic similarity if sentence-transformers available)
_FAQS = [
    {"q": "How do I become an AI/ML Engineer?", "a": "Start with Python, linear algebra, probability, then take ML courses and build projects. Consider the Machine Learning Specialization by Andrew Ng and deep learning courses."},
    {"q": "How to become a UI/UX Designer?", "a": "Learn design principles, prototyping tools (Figma), and create a portfolio with case studies. Consider Google UX Design Certificate and Interaction Design Foundation courses."},
    {"q": "How to prepare for internships?", "a": "Build a GitHub portfolio, solve DS/Algo problems, tailor your CV, and apply early. Use LinkedIn, AngelList, and company career pages."},
]

# Try to enable semantic search if user installed sentence-transformers and faiss
_USE_EMBEDDING = False
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss
    _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    _faq_texts = [f["q"] + " " + f["a"] for f in _FAQS]
    _faq_embeddings = _embed_model.encode(_faq_texts, convert_to_numpy=True)
    _faiss_index = faiss.IndexFlatIP(_faq_embeddings.shape[1])
    faiss.normalize_L2(_faq_embeddings)
    _faiss_index.add(_faq_embeddings)
    _USE_EMBEDDING = True
    print("Semantic chat enabled (sentence-transformers + faiss available)")
except Exception as e:
    _USE_EMBEDDING = False
    # print("Embedding unavailable:", e)

def chat_answer(user_text: str, top_k:int=2) -> str:
    """
    Lightweight chat: if embeddings available, use semantic search over FAQ then synthesize answer.
    Otherwise use simple keyword matching.
    """
    user_text = user_text.strip()
    if not user_text:
        return "Please write your question about careers, resources, or internships."

    # simple keyword match
    for f in _FAQS:
        if any(word.lower() in user_text.lower() for word in f["q"].split()[:3]):
            return f["a"]

    if _USE_EMBEDDING:
        q_emb = _embed_model.encode([user_text], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, idxs = _faiss_index.search(q_emb, top_k)
        best_idx = int(idxs[0][0])
        return _FAQS[best_idx]["a"]

    # fallback generic response
    return "I can help with career suggestions, recommend resources/courses, and pull remote job listings. Ask 'Recommend courses for AI/ML' or 'Show internships for UI/UX'."

if __name__ == "__main__":
    # quick CLI test
    sample = {
        "math_score":7, "programming_score":8, "creativity_score":6, "communication_score":6,
        "problem_solving_score":7, "theory_score":6, "interest_tech":8, "interest_arts":4,
        "test_business":3, "E_vs_I":5, "S_vs_N":5, "T_vs_F":5, "J_vs_P":5
    }
    try:
        print("Predictions for sample input:")
        print(predict_from_dict(sample))
        print("Resources for AI/ML Engineer:")
        print(get_resources_for_careers(["AI/ML Engineer"]))
        print("Fetch remote jobs (sample):")
        print(get_jobs_from_remotive("AI/ML Engineer"))
        print("Chat sample:")
        print(chat_answer("How do I become an AI/ML Engineer?"))
    except Exception as e:
        print("Error (model may not be trained yet):", e)
