"""V5.5: real production canary. Mevcut 158 testi değiştirmez."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.quality import last_tts, record_tts, reset_quality
from agents.shadow import last_shadow, schedule_shadow
from agents.validator import validate_response
from llm import (
    QWEN_27B,
    bind_canary_model,
    canary_assigns_qwen,
    canary_bucket,
    canary_enabled,
    canary_percent,
    chat_model,
    ollama_chat_payload,
    reasoning_model,
    reasoning_think,
    shadow_model,
    thinking_from_ollama,
    user_content_from_ollama,
    user_visible_text,
)

ROOT = Path(__file__).resolve().parent.parent
BACKEND = Path(__file__).resolve().parent
GERMANY_Q = (
    "Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. Fransa'ya mı yönelmeliyim?"
)
GERMANY_GOLDEN = (
    "Almanya'daki üç potansiyel müşteriyi hemen bırakmazdım. "
    "Fiyat baskısı senin gözlemin; henüz ölçülmüş marj verisi değil. "
    "Önce bu üç lead'de fiyatın gerçekten kabul edilemez marja inip inmediğini ölçelim. "
    "Üçünde de aynı tablo çıkarsa Fransa'yı ikinci test pazarı olarak açarız. "
    "Kararı değiştiren veri: üçünde de marj eşiğinin altına inmesi. "
    "Sonraki adım: Nordöl Import ve Gastro Nord için teklif ve hacmi netleştirmek."
)


def _clear() -> None:
    os.environ.pop("MODEL_CANARY_ENABLED", None)
    os.environ.pop("MODEL_CANARY_PERCENT", None)
    os.environ.pop("MODEL_CANARY_TARGET", None)
    os.environ.pop("OLLAMA_REASONING_MODEL", None)
    os.environ.pop("OLLAMA_SHADOW_MODEL", None)


class V55CanaryConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        _clear()

    def tearDown(self) -> None:
        _clear()

    def test_canary_off_uses_llama3(self) -> None:
        os.environ["MODEL_CANARY_ENABLED"] = "false"
        os.environ["MODEL_CANARY_PERCENT"] = "10"
        self.assertFalse(canary_enabled())
        self.assertFalse(canary_assigns_qwen("any"))
        self.assertIsNone(bind_canary_model("any"))
        self.assertEqual(reasoning_model(), chat_model())
        self.assertNotIn("qwen", reasoning_model().casefold())

    def test_canary_10_session_split(self) -> None:
        os.environ["MODEL_CANARY_ENABLED"] = "true"
        os.environ["MODEL_CANARY_PERCENT"] = "10"
        qwen = llama = 0
        for i in range(1000):
            if canary_assigns_qwen(f"v55-split-{i}"):
                qwen += 1
            else:
                llama += 1
        self.assertGreaterEqual(qwen, 50)
        self.assertLessEqual(qwen, 150)
        self.assertEqual(qwen + llama, 1000)
        sid = "v55-stable"
        first = canary_assigns_qwen(sid)
        for _ in range(30):
            self.assertEqual(canary_assigns_qwen(sid), first)
            self.assertEqual(canary_bucket(sid), canary_bucket(sid))

    def test_canary_100_qwen(self) -> None:
        os.environ["MODEL_CANARY_ENABLED"] = "true"
        os.environ["MODEL_CANARY_PERCENT"] = "100"
        self.assertEqual(bind_canary_model("x"), QWEN_27B)
        self.assertTrue(canary_assigns_qwen("x"))

    def test_canary_rollback_off_no_qwen(self) -> None:
        os.environ["MODEL_CANARY_ENABLED"] = "true"
        os.environ["MODEL_CANARY_PERCENT"] = "100"
        self.assertEqual(bind_canary_model("x"), QWEN_27B)
        os.environ["MODEL_CANARY_ENABLED"] = "false"
        os.environ["MODEL_CANARY_PERCENT"] = "0"
        os.environ["OLLAMA_REASONING_MODEL"] = "llama3"
        self.assertFalse(canary_assigns_qwen("x"))
        self.assertIsNone(bind_canary_model("x"))
        self.assertEqual(reasoning_model(), "llama3")

    def test_shadow_disabled_when_canary_on(self) -> None:
        reset_quality()
        os.environ["MODEL_CANARY_ENABLED"] = "true"
        os.environ["MODEL_CANARY_PERCENT"] = "10"
        os.environ["OLLAMA_SHADOW_MODEL"] = QWEN_27B
        self.assertEqual(shadow_model(), "")
        schedule_shadow(
            http=object(),
            question="q",
            user="u",
            system="s",
            production_path="LLM_CMO_SUCCESS",
            production_text="prod",
        )
        self.assertFalse(last_shadow().get("enabled"))

    def test_production_default_still_llama3(self) -> None:
        _clear()
        self.assertEqual(reasoning_model(), chat_model())
        self.assertNotIn("qwen3.5", (os.getenv("OLLAMA_REASONING_MODEL") or "").casefold())
        self.assertFalse(reasoning_think())
        payload = ollama_chat_payload(QWEN_27B, [{"role": "user", "content": "x"}])
        self.assertEqual(payload["options"]["num_predict"], 280)
        self.assertIs(payload["think"], False)

    def test_user_response_hides_reasoning_model(self) -> None:
        os.environ["MODEL_CANARY_ENABLED"] = "true"
        os.environ["MODEL_CANARY_PERCENT"] = "100"
        self.assertEqual(bind_canary_model("s"), QWEN_27B)
        self.assertEqual(chat_model(), "llama3")
        desk = (ROOT / "components" / "consult-desk.tsx").read_text(encoding="utf-8")
        self.assertNotIn("result.chat_model", desk)
        hist = (ROOT / "components" / "consult-history.tsx").read_text(encoding="utf-8")
        self.assertNotIn("chat_model", hist)

    def test_thinking_never_user_or_tts(self) -> None:
        data = {"message": {"content": "Marjı ölçelim.", "thinking": "SECRET PLAN"}}
        visible = user_visible_text(
            user_content_from_ollama(data), thinking_from_ollama(data)
        )
        self.assertEqual(visible, "Marjı ölçelim.")
        self.assertNotIn("SECRET", visible)
        tts = (BACKEND / "tts_local.py").read_text(encoding="utf-8")
        self.assertNotIn("message.thinking", tts)

    def test_germany_france_golden_first_shot(self) -> None:
        verdict = validate_response(GERMANY_GOLDEN, question=GERMANY_Q)
        self.assertEqual(verdict.status, "PASS")
        low = GERMANY_GOLDEN.casefold()
        self.assertIn("bırakmaz", low)
        self.assertIn("gözlem", low)
        self.assertIn("ikinci", low)
        self.assertNotIn("fransa kesin", low)

    def test_tts_telemetry_no_raw_text(self) -> None:
        reset_quality()
        record_tts({"latency_s": 14.4, "engine": "xtts", "bytes": 1000, "text": "secret"})
        self.assertEqual(last_tts().get("latency_s"), 14.4)
        self.assertNotIn("text", last_tts())
        self.assertNotIn("thinking", last_tts())

    def test_first_shot_not_counted_as_retry(self) -> None:
        from agents.quality import ComposeTrace, snapshot

        reset_quality()
        from agents.quality import record_trace

        record_trace(
            ComposeTrace(
                text="ok",
                path="LLM_CMO_RETRY_SUCCESS",
                direct_status="REJECT",
                retry_status="PASS",
            )
        )
        snap = snapshot()
        self.assertEqual(snap.llm_direct_success_rate, 0.0)
        self.assertEqual(snap.llm_retry_success_rate, 1.0)
        self.assertEqual(snap.real_llm_success_rate, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
