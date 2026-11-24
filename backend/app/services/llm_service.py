import httpx

from app.core.config import settings


class OllamaService:
    def __init__(self):
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model
        self.timeout = settings.ollama_timeout

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
    ) -> str:

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    url,
                    json=payload,
                )

                response.raise_for_status()

                result = response.json()

                answer = result.get("response", "").strip()

                if not answer:
                    raise RuntimeError(
                        "Ollama returned an empty response."
                    )

                return answer

        except httpx.ConnectError as exc:
            raise RuntimeError(
                "Cannot connect to Ollama. "
                "Make sure Ollama is running."
            ) from exc

        except httpx.TimeoutException as exc:
            raise RuntimeError(
                "Ollama request timed out."
            ) from exc

        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama returned HTTP {exc.response.status_code}."
            ) from exc


ollama_service = OllamaService()