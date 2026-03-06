from locust import HttpUser, task, between
from uuid import uuid4

class AdmissionUser(HttpUser):
    # Пользователь думает 3–6 секунд перед новым вопросом
    wait_time = between(3, 6)

    @task
    def ask_question(self):
        payload = {
            "query": "Сколько баллов нужно на прикладную информатику",
            "user_id": str(uuid4) 
        }

        self.client.post(
            "/api/generate/",
            json=payload,
            timeout=120
        )