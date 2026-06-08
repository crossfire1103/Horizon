import asyncio

from src.ai.client import AIClient, TracedAIClient
from src.models import ArtifactConfig
from src.storage.artifacts import ArtifactStore, ai_trace_context


class DummyClient(AIClient):
    def __init__(self):
        self.calls = 0
        self.config = type(
            "Config",
            (),
            {
                "provider": type("Provider", (), {"value": "dummy"})(),
                "model": "dummy-model",
                "temperature": 0.1,
                "max_tokens": 128,
            },
        )()

    async def complete(self, system, user, temperature=None, max_tokens=None):
        self.calls += 1
        return '{"ok": true}'


def test_artifact_store_writes_stage_and_ai_call(tmp_path):
    store = ArtifactStore.from_config(
        ArtifactConfig(enabled=True, root_dir=str(tmp_path), cache_ai_calls=True)
    )

    assert store is not None
    store.save_stage("raw", [{"title": "hello"}])
    with ai_trace_context(stage="analyze", item_id="item-1"):
        store.record_ai_call({"request": {"system": "s", "user": "u"}, "response": "r"})

    assert (store.run_dir / "stages" / "raw.json").exists()
    assert (store.run_dir / "ai_calls.jsonl").exists()


def test_traced_ai_client_replays_cached_response(tmp_path):
    store = ArtifactStore.from_config(
        ArtifactConfig(
            enabled=True,
            root_dir=str(tmp_path),
            cache_ai_calls=True,
            replay_ai_calls=True,
        )
    )
    assert store is not None

    inner = DummyClient()
    traced = TracedAIClient(inner, store)
    first = asyncio.run(traced.complete(system="s", user="u"))
    second = asyncio.run(traced.complete(system="s", user="u"))

    assert first == '{"ok": true}'
    assert second == '{"ok": true}'
    assert inner.calls == 1
