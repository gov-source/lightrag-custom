"""
End-to-end test: memory persistence across sessions.

Run after server restart:
    python tests/test_memory_persistence.py

The test simulates two browser sessions:
  Session A: user states a fact
  Session B: user asks about that fact (no conversation_history sent)

Pass = LLM answer in Session B contains the remembered value.
"""

import json
import sys
import time
import urllib.request
import urllib.error

BASE_URL = "http://localhost:9621"
HEADERS = {"Content-Type": "application/json"}


def query(text: str, conversation_history: list | None = None) -> str:
    payload = {
        "query": text,
        "mode": "hybrid",
        "stream": False,
        "conversation_history": conversation_history or [],
    }
    req = urllib.request.Request(
        f"{BASE_URL}/query",
        data=json.dumps(payload).encode(),
        headers=HEADERS,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read())
            return body.get("response", "")
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.read().decode()[:300]}")
        return ""
    except Exception as e:
        print(f"  Error: {e}")
        return ""


def clear_memory() -> None:
    import urllib.request
    req = urllib.request.Request(
        f"{BASE_URL}/query",  # just a ping to check server alive
        data=json.dumps({"query": "ping", "mode": "naive", "stream": False}).encode(),
        headers=HEADERS,
        method="POST",
    )
    # Directly clear memory file
    import os
    mem_path = "rag_storage/conversation_memory.json"
    if os.path.exists(mem_path):
        with open(mem_path, "w") as f:
            json.dump({"messages": []}, f)
        print("  Memory file cleared.")


def check_server() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5):
            return True
    except Exception:
        return False


def run_test(name: str, fact_message: str, question: str, expected_keyword: str) -> bool:
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

    print(f"\n[Session A] Stating fact: {fact_message[:70]}")
    resp_a = query(fact_message)
    print(f"  Response: {resp_a[:150]}")

    # Small pause to let async writes settle
    time.sleep(1)

    print(f"\n[Session B] Asking (fresh session, no history): {question}")
    resp_b = query(question, conversation_history=[])
    print(f"  Response: {resp_b[:300]}")

    passed = expected_keyword.lower() in resp_b.lower()
    status = "PASS ✓" if passed else "FAIL ✗"
    print(f"\n  Expected keyword '{expected_keyword}' in response: {status}")
    if not passed:
        print(f"  Full response:\n    {resp_b[:500]}")
    return passed


def main():
    print("Memory Persistence Test")
    print("=" * 60)

    if not check_server():
        print("ERROR: Server not reachable at http://localhost:9621")
        print("  Start with: lightrag-server")
        sys.exit(1)
    print("Server reachable ✓")

    print("\nClearing memory before tests...")
    clear_memory()
    time.sleep(0.5)

    results = []

    # Test 1: basic fact recall
    results.append(run_test(
        name="Embedding model name and dimensions",
        fact_message="My embedding model is named AquaVec. It produces 384-dimensional vectors and was trained on scientific literature.",
        question="What vector dimensions does our embedding model produce?",
        expected_keyword="384",
    ))

    time.sleep(1)

    # Test 2: configuration recall
    results.append(run_test(
        name="Deployment configuration recall",
        fact_message="For the AquaVec deployment, we set the batch size to 64 and the max sequence length to 512 tokens.",
        question="What were the inference configuration choices we settled on?",
        expected_keyword="64",
    ))

    time.sleep(1)

    # Test 3: model name recall — AquaVec is already in memory from test 1; this
    # tests that retrieval works on a *different* question phrasing too.
    results.append(run_test(
        name="Model name recall (different phrasing)",
        fact_message="The project codename for our embedding stack is Nebula.",
        question="What is the name of our embedding model?",
        expected_keyword="AquaVec",
    ))

    print(f"\n{'='*60}")
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    if passed == total:
        print("ALL TESTS PASSED — memory persistence is working correctly.")
    else:
        print("SOME TESTS FAILED — check server logs for [memory] lines.")
        print("  Tip: grep for '[memory]' in server terminal output")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
