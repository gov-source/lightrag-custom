import json
from uuid import uuid4

import numpy as np
import pytest

from lightrag.base import QueryParam
from lightrag.lightrag import LightRAG
from lightrag.utils import EmbeddingFunc


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_conversation_history_is_persisted_and_retrieved(tmp_path):
    workspace = f"memory_test_{uuid4().hex[:8]}"
    rag = LightRAG(
        working_dir=str(tmp_path),
        workspace=workspace,
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=8192,
            func=_dummy_embedding,
        ),
    )
    await rag.initialize_storages()

    query_history = [
        {"role": "user", "content": "Hello, this is a memory test message."}
    ]
    param = QueryParam(conversation_history=query_history)

    await rag._persist_conversation_history_memory(param)

    conversation_memory_path = rag._conversation_memory_path
    assert conversation_memory_path is not None

    with open(conversation_memory_path, encoding="utf-8") as f:
        memory_data = json.load(f)

    assert isinstance(memory_data, dict)
    assert "messages" in memory_data
    assert memory_data["messages"][0]["content"] == query_history[0]["content"]

    retrieved = await rag._retrieve_conversation_memory("Tell me about the memory test", top_k=1)
    assert len(retrieved) == 1
    assert retrieved[0]["role"] == query_history[0]["role"]
    assert retrieved[0]["content"] == query_history[0]["content"]
