# import json
# import faiss
# import os
# import numpy as np
# from sentence_transformers import SentenceTransformer
#
# # === Пути ===
# INDEX_PATH = "data-older/all_faiss_index.idx"
# META_PATH = "data-older/all_documents_meta.json"
#
# model = SentenceTransformer('../sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
#
# index = faiss.read_index(INDEX_PATH)
# with open(META_PATH, "r", encoding="utf-8") as f:
#     metadata = json.load(f)
#
# # Новые тексты
# source = 'пояснения'
# text = 'Начальник УКиМ - Головей Станислав Игоревич'
# summarize = 'Начальник УКиМ - Головей Станислав Игоревич'
# full_text = {"source": source, "text": text, "summarize": summarize}
#
# embedding = model.encode([full_text["summarize"]]).astype("float32")
#
# # === Шаг 4: Добавляем в индекс и метаданные ===
# index.add(embedding)
# metadata.append(full_text)
#
# # === Шаг 5: Сохраняем ===
# faiss.write_index(index, INDEX_PATH)
#
# with open(META_PATH, "w", encoding="utf-8") as f:
#     json.dump(metadata, f, ensure_ascii=False, indent=2)
#
# print("✅ Добавлено в индекс и сохранено.")


import requests
import json


def test_update_index():
    """Тестирование обновления FAISS индекса через API"""

    # URL вашего сервера
    base_url = "https://ragbot.loca.lt"

    # Данные для добавления
    new_document = {
        "source": "пояснения",
        "text": "Начальник УКиМ - Головей Станислав Игоревич",
        "summarize": "Начальник УКиМ - Головей Станислав Игоревич"
    }

    try:
        # 1. Получаем текущую статистику
        print("📊 Получение текущей статистики...")
        stats_response = requests.get(f"{base_url}/index-stats")
        if stats_response.status_code == 200:
            stats = stats_response.json()
            print(f"Текущее количество документов: {stats.get('total_documents', 'N/A')}")
        else:
            print(f"Ошибка получения статистики: {stats_response.status_code}")

        # 2. Добавляем новый документ
        print("\n📝 Добавление нового документа...")
        update_response = requests.post(
            f"{base_url}/update-index",
            json=new_document,
            headers={"Content-Type": "application/json"}
        )

        if update_response.status_code == 200:
            result = update_response.json()
            print("✅ Документ успешно добавлен!")
            print(f"Сообщение: {result['message']}")
            print(f"Всего документов: {result['total_documents']}")
        else:
            print(f"❌ Ошибка добавления документа: {update_response.status_code}")
            print(f"Ответ: {update_response.text}")

        # 3. Проверяем обновленную статистику
        print("\n📊 Проверка обновленной статистики...")
        stats_response = requests.get(f"{base_url}/index-stats")
        if stats_response.status_code == 200:
            stats = stats_response.json()
            print(f"Новое количество документов: {stats.get('total_documents', 'N/A')}")
            print(f"Размер индекса: {stats.get('index_size', 'N/A')}")
            print(f"Размерность векторов: {stats.get('index_dimension', 'N/A')}")
        else:
            print(f"Ошибка получения статистики: {stats_response.status_code}")

    except requests.exceptions.ConnectionError:
        print("❌ Не удалось подключиться к серверу. Убедитесь, что сервер запущен на localhost:8000")
    except Exception as e:
        print(f"❌ Произошла ошибка: {e}")


def test_multiple_documents():
    """Тестирование добавления нескольких документов"""

    base_url = "http://localhost:8000"

    documents = [
        {
            "source": "контакты",
            "text": "Приемная комиссия работает с 9:00 до 17:00",
            "summarize": "Время работы приемной комиссии: 9:00-17:00"
        },
        {
            "source": "правила",
            "text": "Подача документов осуществляется только в электронном виде",
            "summarize": "Документы подаются в электронном виде"
        },
        {
            "source": "требования",
            "text": "Минимальный проходной балл по математике - 75",
            "summarize": "Проходной балл математика: 75"
        }
    ]

    print("🔄 Добавление нескольких документов...")

    for i, doc in enumerate(documents, 1):
        try:
            response = requests.post(
                f"{base_url}/update-index",
                json=doc,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Документ {i}/3 добавлен. Всего документов: {result['total_documents']}")
            else:
                print(f"❌ Ошибка добавления документа {i}: {response.status_code}")

        except Exception as e:
            print(f"❌ Ошибка при добавлении документа {i}: {e}")


if __name__ == "__main__":
    print("🚀 Тестирование API обновления FAISS индекса")
    print("=" * 50)

    # Тест добавления одного документа
    test_update_index()

    print("\n" + "=" * 50)

    # Тест добавления нескольких документов
    # test_multiple_documents()

    print("\n✅ Тестирование завершено!")