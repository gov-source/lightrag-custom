import json
import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator

from dotenv import load_dotenv

from lightrag.utils import logger


_TRUTHY = {"1", "true", "yes", "on"}


def _env_enabled(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in _TRUTHY


def _configure_langfuse_env() -> None:
    load_dotenv(dotenv_path=".env", override=False)
    if not os.getenv("LANGFUSE_HOST") and os.getenv("LANGFUSE_BASE_URL"):
        os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_BASE_URL", "")


@lru_cache(maxsize=1)
def _get_langfuse_client() -> Any | None:
    _configure_langfuse_env()
    if not _env_enabled("LANGFUSE_ENABLE_TRACE"):
        return None
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        logger.warning(
            "LANGFUSE_ENABLE_TRACE=true but Langfuse credentials are incomplete"
        )
        return None

    try:
        from langfuse import get_client

        return get_client()
    except Exception as exc:
        logger.warning(f"Langfuse tracing disabled: {exc}")
        return None


def _truncate(value: Any, max_chars: int) -> Any:
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[:max_chars] + "...[truncated]"
    if isinstance(value, list):
        return [_truncate(item, max_chars) for item in value]
    if isinstance(value, dict):
        return {key: _truncate(item, max_chars) for key, item in value.items()}
    return value


def _capture_content() -> bool:
    return _env_enabled("LANGFUSE_CAPTURE_CONTENT", "false")


def should_capture_content() -> bool:
    return _capture_content()


def _max_content_chars() -> int:
    try:
        return max(0, int(os.getenv("LANGFUSE_MAX_CONTENT_CHARS", "4000")))
    except ValueError:
        return 4000


def build_llm_trace_input(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    history_messages = history_messages or []
    if not _capture_content():
        return {
            "prompt_chars": len(prompt or ""),
            "system_prompt_chars": len(system_prompt or ""),
            "history_messages": len(history_messages),
            "content_capture": "disabled",
        }

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    return {"messages": _truncate(messages, _max_content_chars())}


def build_embedding_trace_input(texts: list[str]) -> dict[str, Any]:
    if not _capture_content():
        return {
            "text_count": len(texts),
            "total_chars": sum(len(text or "") for text in texts),
            "content_capture": "disabled",
        }
    return {"texts": _truncate(texts, _max_content_chars())}


def build_trace_output(output: Any) -> Any:
    if not _capture_content():
        if isinstance(output, str):
            return {"output_chars": len(output)}
        if isinstance(output, list):
            return {"item_count": len(output)}
        return None
    return _truncate(output, _max_content_chars())


def ollama_usage_details(response: Any) -> dict[str, int]:
    usage: dict[str, int] = {}
    if isinstance(response, dict):
        prompt_tokens = response.get("prompt_eval_count")
        completion_tokens = response.get("eval_count")
    else:
        prompt_tokens = getattr(response, "prompt_eval_count", None)
        completion_tokens = getattr(response, "eval_count", None)

    if isinstance(prompt_tokens, int):
        usage["input_tokens"] = prompt_tokens
    if isinstance(completion_tokens, int):
        usage["output_tokens"] = completion_tokens
    return usage


@contextmanager
def langfuse_trace(
    *,
    name: str,
    input: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any | None]:
    """Root trace context.  All langfuse_observation calls made inside an async
    coroutine that awaits within this block will be nested under this span,
    giving one unified trace per high-level operation (query / insert)."""
    langfuse = _get_langfuse_client()
    if langfuse is None:
        yield None
        return

    kwargs: dict[str, Any] = {"name": name, "as_type": "span"}
    if input is not None:
        kwargs["input"] = input
    if metadata:
        kwargs["metadata"] = metadata

    try:
        observation_cm = langfuse.start_as_current_observation(**kwargs)
    except Exception as exc:
        logger.debug(f"Failed to start Langfuse root trace: {exc}")
        yield None
        return

    with observation_cm as trace:
        yield trace


@contextmanager
def langfuse_observation(
    *,
    name: str,
    as_type: str,
    model: str | None = None,
    input: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any | None]:
    langfuse = _get_langfuse_client()
    if langfuse is None:
        yield None
        return

    kwargs: dict[str, Any] = {
        "name": name,
        "as_type": as_type,
    }
    if model:
        kwargs["model"] = model
    if input is not None:
        kwargs["input"] = input
    if metadata:
        kwargs["metadata"] = metadata

    try:
        observation_cm = langfuse.start_as_current_observation(**kwargs)
    except Exception as exc:
        logger.debug(f"Failed to start Langfuse observation: {exc}")
        yield None
        return

    with observation_cm as observation:
        yield observation


def update_observation(observation: Any | None, **kwargs: Any) -> None:
    if observation is None:
        return
    try:
        observation.update(**{key: value for key, value in kwargs.items() if value is not None})
    except Exception as exc:
        logger.debug(f"Failed to update Langfuse observation: {exc}")


def safe_json_metadata(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)
