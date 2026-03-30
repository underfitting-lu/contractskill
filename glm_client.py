import os

from env.api_env import load_api_env_file


class ModelRequestError(RuntimeError):
    def __init__(
        self,
        *,
        error_code: str,
        status_code: int | None,
        provider_message: str,
        raw_error: str,
    ) -> None:
        self.error_code = error_code
        self.status_code = status_code
        self.provider_message = provider_message
        self.raw_error = raw_error
        super().__init__(f"{error_code}: {provider_message}")


def _clean_env_value(value: str | None) -> str:
    text = (value or "").replace("\r", "").replace("\n", "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return text


def _validate_api_key(name: str, value: str | None) -> str:
    key = _clean_env_value(value)
    if not key:
        raise RuntimeError(f"{name} is not set.")
    if not key.isascii():
        raise RuntimeError(
            f"{name} contains non-ASCII characters. "
            "This usually means a placeholder like '???key' was copied literally."
        )

    lowered = key.lower()
    placeholder_markers = ("your", "newkey", "realkey", "placeholder", "example")
    if any(marker in lowered for marker in placeholder_markers):
        raise RuntimeError(
            f"{name} looks like a placeholder value, not a real API key."
        )
    return key


def _extract_status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value

    response = getattr(exc, "response", None)
    if response is not None:
        for attr in ("status_code", "status"):
            value = getattr(response, attr, None)
            if isinstance(value, int):
                return value
    return None


def _extract_error_text(exc: Exception) -> str:
    parts: list[str] = []
    for attr in ("message", "body"):
        value = getattr(exc, attr, None)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    response = getattr(exc, "response", None)
    if response is not None:
        for attr in ("text",):
            value = getattr(response, attr, None)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
    parts.append(str(exc).strip())
    seen: set[str] = set()
    deduped: list[str] = []
    for item in parts:
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return " | ".join(deduped)


def _classify_model_error(exc: Exception) -> ModelRequestError:
    status_code = _extract_status_code(exc)
    raw_error = _extract_error_text(exc)
    lowered = raw_error.lower()

    error_code = "infra_model_request_failed"
    if status_code == 401 or "authentication" in lowered or "invalid api key" in lowered:
        error_code = "infra_api_401"
    elif status_code == 429 or "rate limit" in lowered or "quota" in lowered:
        error_code = "infra_api_429"
    elif "auth_unavailable" in lowered:
        error_code = "infra_auth_unavailable"
    elif status_code is not None and status_code >= 500:
        error_code = f"infra_api_{status_code}"
    elif "temporarily unavailable" in lowered or "service unavailable" in lowered:
        error_code = "infra_api_unavailable"
    elif "connection" in lowered or "timeout" in lowered:
        error_code = "infra_api_connection"

    provider_message = raw_error or "Model request failed."
    return ModelRequestError(
        error_code=error_code,
        status_code=status_code,
        provider_message=provider_message,
        raw_error=raw_error,
    )


class GLMClient:
    def __init__(self, model: str = "glm-4.6v"):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai is not installed in the current environment. "
                "Install it in the VWA virtualenv before running the benchmark."
            ) from exc

        load_api_env_file()
        self.model = model
        api_key = _validate_api_key("ZAI_API_KEY", os.getenv("ZAI_API_KEY"))
        base_url = _clean_env_value(
            os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
        )
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def chat_completion(self, **kwargs):
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise _classify_model_error(exc) from exc

    def ask_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        response = self.chat_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def ask_vision(self, text_prompt: str, image_url: str, temperature: float = 0.0) -> str:
        response = self.chat_completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
