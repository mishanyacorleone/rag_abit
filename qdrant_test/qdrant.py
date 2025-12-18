from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from uuid import uuid4
import json
import os

# os.environ['HTTP_PROXY'] = "socks5://localhost:7113"
# os.environ['HTTPS_PROXY'] = "socks5://localhost:7113"
# os.environ['NO_PROXY'] = "localhost,127.0.0.1,0.0.0.0"

client = QdrantClient("localhost", port=6333)
# client = QdrantClient(":memory:")

model = SentenceTransformer(
    model_name_or_path="/mnt/mishutqa/PycharmProjects/abitBot/app/models/deepvk/USER-bge-m3",
    local_files_only=True,
    device='cuda'
)

# with open("all_documents_meta - копия.json", "r", encoding='utf-8') as file:
#     data = json.load(file)
#
#
# client.create_collection(
#     collection_name="documents",
#     vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
# )
#
#
# for doc in data:
#     embeddings = model.encode(doc["text"])
#
#     client.upsert(
#         collection_name="documents",
#         points=[
#             PointStruct(
#                 id=str(uuid4()),
#                 vector=embeddings,
#                 payload={
#                     "text": doc["text"]
#                 }
#             )
#         ]
#     )


def search(query: str, top_k: int = 3):
    query_vector = model.encode(query).tolist()
    results = client.query_points(
        collection_name="documents",
        query=query_vector,
        limit=3
    )
    return results


if __name__ == "__main__":
    print("\n🔍 Qdrant поиск готов! Введите ваш запрос (или 'exit' для выхода):")
    while True:
        query = input("\nВаш запрос: ").strip()
        if query.lower() in ("exit", "quit", "выход"):
            break
        if not query:
            continue

        hits = search(query).points
        for i, hit in enumerate(hits):
            print(f"{i}.\n")
            print(f"Текст: {hit.payload['text']}")
            print(f"Score: {hit.score}")