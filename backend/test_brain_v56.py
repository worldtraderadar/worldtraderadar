"""V5.6 production readiness. Mevcut 169 testi değiştirmez."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.quality import ComposeTrace, record_trace, reset_quality, snapshot
from agents.reject_class import classify_reject_cause, looks_truncated
from agents.response_engine import retry_draft_for, speakable_instruction
from agents.validator import validate_response
from llm import (
    QWEN_27B,
    QWEN_NUM_PREDICT,
    bind_canary_model,
    canary_assigns_qwen,
    canary_bucket,
    canary_enabled,
    canary_percent,
    canary_unload_peer,
    chat_model,
    ollama_chat_payload,
    peer_reasoning_model,
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
    os.environ.pop("MODEL_CANARY_UNLOAD_PEER", None)
    os.environ.pop("OLLAMA_REASONING_MODEL", None)
    os.environ.pop("OLLAMA_SHADOW_MODEL", None)


def _split(prefix: str, n: int = 1000) -> tuple[int, int]:
    qwen = sum(1 for i in range(n) if canary_assigns_qwen(f"{prefix}-{i}"))
    return qwen, n - qwen


class V56CanaryLadderTests(unittest.TestCase):
    def setUp(self) -> None:
        _clear()

    def tearDown(self) -> None:
        _clear()

    def test_hash_routing_unchanged(self) -> None:
        os.environ["MODEL_CANARY_ENABLED"] = "true"
        os.environ["MODEL_CANARY_PERCENT"] = "10"
        self.assertEqual(canary_bucket("v55-live-0"), 25)
        self.assertEqual(canary_bucket("v55-live-10"), 0)
        self.assertFalse(canary_assigns_qwen("v55-live-0"))
        self.assertTrue(canary_assigns_qwen("v55-live-10"))

    def test_canary_25_split_and_session_sticky(self) -> None:
        os.environ["MODEL_CANARY_ENABLED"] = "true"
        os.environ["MODEL_CANARY_PERCENT"] = "25"
        qwen, llama = _split("v56-p25")
        self.assertGreaterEqual(qwen, 180)
        self.assertLessEqual(qwen, 320)
        self.assertEqual(qwen + llama, 1000)
        sid = "v56-sticky-25"
        first = canary_assigns_qwen(sid)
        for _ in range(40):
            self.assertEqual(canary_assigns_qwen(sid), first)
            self.assertEqual(canary_bucket(sid), canary_bucket(sid))

    def test_canary_50_split(self) -> None:
        os.environ["MODEL_CANARY_ENABLED"] = "true"
        os.environ["MODEL_CANARY_PERCENT"] = "50"
        qwen, llama = _split("v56-p50")
        self.assertGreaterEqual(qwen, 420)
        self.assertLessEqual(qwen, 580)
        self.assertEqual(qwen + llama, 1000)

    def test_canary_100_explicit_qwen_path_keeps_llama_default(self) -> None:
        os.environ["MODEL_CANARY_ENABLED"] = "true"
        os.environ["MODEL_CANARY_PERCENT"] = "100"
        self.assertEqual(bind_canary_model("any"), QWEN_27B)
        self.assertTrue(canary_assigns_qwen("any"))
        self.assertEqual(reasoning_model(), chat_model())
        self.assertNotIn("qwen3.5", reasoning_model().casefold())

    def test_rollback_25_50_100_to_zero(self) -> None:
        os.environ["MODEL_CANARY_ENABLED"] = "true"
        for pct in ("25", "50", "100"):
            os.environ["MODEL_CANARY_PERCENT"] = pct
            self.assertTrue(canary_enabled())
            os.environ["MODEL_CANARY_PERCENT"] = "0"
            self.assertFalse(canary_assigns_qwen("v56-roll"))
            self.assertIsNone(bind_canary_model("v56-roll"))
            os.environ["MODEL_CANARY_PERCENT"] = pct
        os.environ["MODEL_CANARY_ENABLED"] = "false"
        os.environ["MODEL_CANARY_PERCENT"] = "0"
        self.assertFalse(canary_assigns_qwen("v56-roll"))
        self.assertEqual(reasoning_model(), chat_model())

    def test_shadow_stays_off_during_canary(self) -> None:
        os.environ["MODEL_CANARY_ENABLED"] = "true"
        os.environ["MODEL_CANARY_PERCENT"] = "25"
        os.environ["OLLAMA_SHADOW_MODEL"] = QWEN_27B
        self.assertEqual(shadow_model(), "")

    def test_production_default_not_auto_qwen(self) -> None:
        _clear()
        self.assertEqual(reasoning_model(), "llama3")
        self.assertFalse(reasoning_think())
        self.assertEqual(QWEN_NUM_PREDICT, 280)
        payload = ollama_chat_payload(QWEN_27B, [{"role": "user", "content": "x"}])
        self.assertEqual(payload["options"]["num_predict"], 280)
        self.assertIs(payload["think"], False)
        env_local = (ROOT / ".env.local").read_text(encoding="utf-8")
        self.assertNotIn("OLLAMA_REASONING_MODEL=qwen", env_local.casefold())


class V56LifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        _clear()

    def tearDown(self) -> None:
        _clear()

    def test_unload_peer_default_off(self) -> None:
        os.environ["MODEL_CANARY_ENABLED"] = "true"
        os.environ["MODEL_CANARY_PERCENT"] = "10"
        self.assertFalse(canary_unload_peer())

    def test_unload_peer_requires_canary(self) -> None:
        os.environ["MODEL_CANARY_UNLOAD_PEER"] = "true"
        self.assertFalse(canary_unload_peer())
        os.environ["MODEL_CANARY_ENABLED"] = "true"
        os.environ["MODEL_CANARY_PERCENT"] = "10"
        self.assertTrue(canary_unload_peer())

    def test_peer_model_llama_qwen(self) -> None:
        self.assertEqual(peer_reasoning_model("llama3"), QWEN_27B)
        self.assertEqual(peer_reasoning_model(QWEN_27B), "llama3")
        self.assertIsNone(peer_reasoning_model(""))


class V56ValidatorAndCompletenessTests(unittest.TestCase):
    def test_validator_not_loosened_unsupported_percent(self) -> None:
        self.assertEqual(
            validate_response("Almanya pazarı yüzde 12 büyüdü.", question=GERMANY_Q).status,
            "REJECT",
        )

    def test_validator_not_loosened_false_france(self) -> None:
        text = "Fransa kesin daha kârlı, Almanya'yı bırak."
        self.assertEqual(validate_response(text, question=GERMANY_Q).status, "REJECT")

    def test_validator_not_loosened_memory(self) -> None:
        self.assertEqual(
            validate_response(
                "Hatırlıyorum, geçen sefer marjın yüzde 40'tı.",
                facts="BELLEK BOŞ: hatırlıyorum deme.",
            ).status,
            "REJECT",
        )

    def test_germany_golden_still_pass(self) -> None:
        self.assertEqual(validate_response(GERMANY_GOLDEN, question=GERMANY_Q).status, "PASS")

    def test_retry_draft_truncates_long_length_hits(self) -> None:
        long = ("Marjı ölçmeden kabul etmem. " * 40).strip()
        short = retry_draft_for(long, "length", limit=80)
        self.assertLess(len(short), len(long))
        self.assertTrue(short.endswith("…"))
        same = retry_draft_for("Kısa cevap.", "length")
        self.assertEqual(same, "Kısa cevap.")
        untouched = retry_draft_for(long, "stop")
        self.assertEqual(untouched, long)

    def test_speakable_asks_complete_short_reply(self) -> None:
        text = speakable_instruction()
        self.assertIn("yarım", text.casefold())
        self.assertIn("beş", text.casefold())

    def test_classify_truncation_vs_unsupported(self) -> None:
        cut = "Tahsilat riski yüksek, siparişi şimdi kabul etme çünkü kredi kaydı"
        row = classify_reject_cause(cut, reasons=[], done_reason="length", status="REJECT")
        self.assertTrue(looks_truncated(cut, "length"))
        self.assertEqual(row["code"], "C")
        unsourced = classify_reject_cause(
            "Pazar yüzde 18 büyüdü.",
            reasons=["kaynaksız rakam: yüzde 18"],
            done_reason="stop",
            status="REJECT",
        )
        self.assertEqual(unsourced["code"], "E")
        quality = classify_reject_cause(
            "Fransa kesin daha kârlı.",
            reasons=["kaynaksız mutlak kesinlik"],
            done_reason="stop",
            status="REJECT",
        )
        self.assertEqual(quality["code"], "A")
        self.assertNotEqual(row["code"], "B")

    def test_length_complete_sentence_not_false_truncated(self) -> None:
        done = "Almanya'yı bırakmam. Önce marjı ölçelim."
        self.assertFalse(looks_truncated(done, "length"))

    def test_thinking_never_user(self) -> None:
        data = {"message": {"content": "Marjı ölçelim.", "thinking": "SECRET"}}
        visible = user_visible_text(
            user_content_from_ollama(data), thinking_from_ollama(data)
        )
        self.assertEqual(visible, "Marjı ölçelim.")
        self.assertNotIn("SECRET", visible)

    def test_first_shot_not_retry(self) -> None:
        reset_quality()
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

    def test_voice_stack_files_untouched_contract(self) -> None:
        tts = (BACKEND / "tts_local.py").read_text(encoding="utf-8")
        self.assertNotIn("message.thinking", tts)
        desk = (ROOT / "components" / "consult-desk.tsx").read_text(encoding="utf-8")
        self.assertNotIn("result.chat_model", desk)


if __name__ == "__main__":
    unittest.main(verbosity=2)
