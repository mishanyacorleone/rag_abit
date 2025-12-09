import json
import time


# Функция первичного поиска
def retrieve_from_index(query, index, texts, retriever_model, top_k=10):
    start_time = time.time()
    query_embedding = retriever_model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        results.append({
            "score": float(score),
            "corpus_id": int(idx),
            "candidate": texts[idx]
        })

    print('Затраченное время на первичный поиск:', str(time.time() - start_time))
    return results


# Функция реранка
def rerank_documents(query, candidates, reranker_model, top_k=3):
    # Убедимся, что candidate — это строка
    start_time = time.time()
    pairs = []
    print(candidates)
    for doc in candidates:
        candidate_text = doc["candidate"]
        if isinstance(candidate_text, dict):
            candidate_text = json.dumps(candidate_text, ensure_ascii=False)
        elif not isinstance(candidate_text, str):
            candidate_text = str(candidate_text)
        pairs.append([query, candidate_text])
    # Получаем оценки от CrossEncoder
    scores = reranker_model.predict(pairs)
    # Объединяем оценки с кандидатами
    reranked = []
    for score, doc in zip(scores, candidates):
        reranked.append({
            "score": float(score),
            "corpus_id": doc["corpus_id"],
            "candidate": doc["candidate"]
        })
    # Сортировка по убыванию
    reranked = sorted(reranked, key=lambda x: x["score"], reverse=True)

    print('Затраченное время на реранк:', str(time.time() - start_time))
    return reranked[:top_k]


# Главная функция
def retrieve_documents(query, retriever_model, reranker_model, index, all_documents, top_k=3):
    initial_candidates = retrieve_from_index(query, index, all_documents, retriever_model, top_k=10)
    return rerank_documents(query, initial_candidates, reranker_model, top_k=top_k)
