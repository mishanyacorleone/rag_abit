# import json
# import re
#
# # Загрузка всех источников
# def load_json(file_path):
#     with open(file_path, "r", encoding="utf-8") as f:
#         return json.load(f)
#
# spo_data = load_json("Приложения итог/Приложение №2 новое.json")
# school_data = load_json("Приложения итог/Приложение №3 новое.json")
# indiv_achievements = load_json("Приложения итог/Приложение №9.json")
# pravila_chunks = load_json("data-older/pravila_chunks.json")
# spec_chunks = load_json("data-older/spec_chunks.json")
# result_marks = load_json('Приложения итог/Проходные баллы 2023-2024 год.json')
# date_data = load_json('Приложения итог/Приложение №1.json')
#
# # Преобразование вступительных испытаний (СПО и школа) в строки
#
# def exams_to_text(entry):
#     exams = []
#     for i in range(1, 5):
#         exam = entry.get(f"Вступительное испытание №{i}")
#         if exam and exam != "Нет":
#             exams.append(exam)
#     return f"Код специальности: {entry['Код специальности']}. Направление: {entry['Направление и профиль подготовки']}. Экзамены: {'; '.join(exams)}. Баллы: {entry.get('Минимальное и максимальное количество баллов', 'нет данных')}"
#
# spo_texts = [
#     {"source": "экзамены_спо", "text": exams_to_text(entry)}
#     for entry in spo_data
# ]
#
# school_texts = [
#     {"source": "экзамены_школа", "text": exams_to_text(entry)}
#     for entry in school_data
# ]
#
# # Преобразование достижений
# achievements_texts = [
#     {"source": "индивидуальные_достижения", "text": f"Достижение: {a['Наименование индивидуального достижения']} Подтверждение: {a['Документы, подтверждающие получение индивидуального достижения']}. Баллы: {a['Балл']}"}
#     for a in indiv_achievements
# ]
#
# # Преобразование spec_chunks и pravila_chunks в общий вид
# spec_chunks = [
#     {"source": "специальности", "text": chunk}
#     for chunk in spec_chunks
# ]
# pravila_chunks = [
#     {"source": "правила_приема", "text": chunk}
#     for chunk in pravila_chunks
# ]
#
# # Преобразование Проходных баллов в общий вид
# marks_chunks = [
#     {'source': 'проходные_баллы',
#      'text': f'Проходной балл на направление {spec} в {mark}'}
#     for spec, mark in result_marks.items()
# ]
#
#
# # Объединение всех текстов
# all_documents = spo_texts + school_texts + achievements_texts + spec_chunks + pravila_chunks + marks_chunks
#
# all_documents.append(date_data)
#
# # Сохраняем результат
# with open("all_documents_for_faiss.json", "w", encoding="utf-8") as f:
#     json.dump(all_documents, f, ensure_ascii=False, indent=2)

'''Создание FAISS индекса'''

import json
import faiss
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent.parent
print(PROJECT_DIR)
# exit()

# === 1. Загрузка объединённого файла с документами ===
with open(PROJECT_DIR / "data" / "all_documents_meta.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

texts = [doc['text'] for doc in documents]

# === 2. Создание эмбеддингов ===
model = SentenceTransformer(str(PROJECT_DIR / "app" / "models" / "deepvk" / "USER-bge-m3"), device='cpu')
model.eval()

with torch.no_grad():
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=4).astype("float32")

# === 3. Создание FAISS-индекса ===
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)  # Используем cosine similarity с нормализованными векторами

# Нормализуем вектора
faiss.normalize_L2(embeddings)

# Добавляем векторные представления в индекс
index.add(embeddings)

# === 4. Сохраняем индекс и метаинформацию ===
faiss.write_index(index, str(PROJECT_DIR / "data" / "all_faiss_index.idx"))

with open(PROJECT_DIR / "data" / "all_documents_meta.json", "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)

print("✅ Индекс и метафайл успешно сохранены")
