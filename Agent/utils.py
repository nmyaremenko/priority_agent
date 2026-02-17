import requests
from typing import List, Dict, Any
from config import Config

def get_embedding(text: str, model: str = Config.EMBEDDING_MODEL) -> List[float]:
    """Генерация эмбеддинга для текста"""
    if not model:
        raise ValueError("Embedding model is not configured for current provider")

    client = requests.Session()
    response = client.post(
        f"{Config.LLM_API_URL}/embeddings",
        headers={"Authorization": f"Bearer {Config.API_KEY}"},
        json={
            "model": model,
            "input": text
        }
    )
    if response.status_code == 200:
        return response.json()["data"][0]["embedding"]
    else:
        raise Exception(f"Embedding API error: {response.status_code} - {response.text}")

def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Расчет косинусного сходства между двумя эмбеддингами"""
    import numpy as np
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def normalize_score(score: float, min_val: float = 0, max_val: float = 100) -> float:
    """Нормализация оценки в диапазон от 0 до 1"""
    return max(0, min(1, (score - min_val) / (max_val - min_val)))
