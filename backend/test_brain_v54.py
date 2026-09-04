"""V5.4: shadow isolation, canary routing, user-claim vs fact, production gate.

Mevcut 134 testi değiştirmez. Production default llama3 kalır.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.evaluator import score_cmo_v51
from agents.quality import last_inference, record_inference, reset_quality, snapshot
from agents.quality_flags import is_false_certainty, is_memory_dishonest, is_prompt_leak
from agents.response_engine import compose_consultant_traced
from agents.shadow import last_shadow, run_shadow_eval
from agents.validator import (
    evaluator_unsourced_stats,
    is_user_claim_stat,
    make_speakable,
    validate_response,
)
from llm import (
    QWEN_27B,
    bind_canary_model,
    canary_assigns_qwen,
    canary_enabled,
    canary_percent,
    chat_model,
    ollama_chat_payload,
    reasoning_model,
    reasoning_override,
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
COMP_Q = "Rakibim fiyatı benden %20 düşük."


def _clear_canary() -> None:
    os.environ.pop("MODEL_CANARY_ENABLED", None)
    os.environ.pop("MODEL_CANARY_PERCENT", None)
    os.environ.pop("MODEL_CANARY_TARGET", None)
    os.environ.pop("OLLAMA_REASONING_MODEL", None)
    os.environ.pop("OLLAMA_SHADOW_MODEL", None)


class _Resp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _Http:
    def __init__(self, payload: dict | None = None) -> None:
        self.bodies: list[dict] = []
        self.urls: list[str] = []
        self._payload = payload or {
            "message": {"content": "Shadow içerik", "thinking": "SECRET_THINKING"},
            "done_reason": "stop",
            "eval_count": 8,
            "eval_duration": 400_000_000,
            "prompt_eval_duration": 100_000_000,
            "load_duration": 5_000_000,
        }

    async def post(self, url, json=None, timeout=None):
        self.urls.append(url)
        self.bodies.append(json or {})
        return _Resp(self._payload)


class V54ShadowIsolationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _clear_canary()
        reset_quality()

    def tearDown(self) -> None:
        _clear_canary()

    async def test_shadow_does_not_change_production_response(self) -> None:
        os.environ["OLLAMA_SHADOW_MODEL"] = QWEN_27B
        prod = "Kullanıcıya giden llama cevabı"
        http = _Http()
        await run_shadow_eval(
            http=http,
            question=GERMANY_Q,
            user="u",
            system="s",
            production_path="LLM_CMO_SUCCESS",
            production_text=prod,
        )
        self.assertEqual(prod, "Kullanıcıya giden llama cevabı")
        self.assertEqual(last_shadow()["user_visible"], "Shadow içerik")
        self.assertNotEqual(last_shadow()["user_visible"], prod)
        self.assertIs(http.bodies[0]["think"], False)
        self.assertEqual(http.bodies[0].get("keep_alive"), "0")

    async def test_shadow_does_not_reach_tts(self) -> None:
        os.environ["OLLAMA_SHADOW_MODEL"] = QWEN_27B
        http = _Http()
        await run_shadow_eval(
            http=http,
            question=GERMANY_Q,
            user="u",
            system="s",
            production_path="LLM_CMO_SUCCESS",
            production_text="prod",
        )
        tts = (BACKEND / "tts_local.py").read_text(encoding="utf-8")
        shadow_src = (BACKEND / "agents" / "shadow.py").read_text(encoding="utf-8")
        self.assertNotIn("/speak", "".join(http.urls))
        self.assertNotIn("synthesize", shadow_src)
        self.assertNotIn("tts_local", shadow_src)
        self.assertNotIn("message.thinking", tts)

    async def test_shadow_does_not_write_memory(self) -> None:
        shadow_src = (BACKEND / "agents" / "shadow.py").read_text(encoding="utf-8")
        self.assertNotIn("persist_from_turn", shadow_src)
        self.assertNotIn("persist_memory", shadow_src)
        os.environ["OLLAMA_SHADOW_MODEL"] = QWEN_27B
        http = _Http()
        await run_shadow_eval(
            http=http,
            question=GERMANY_Q,
            user="u",
            system="s",
            production_path="LLM_CMO_SUCCESS",
            production_text="prod",
        )
        self.assertNotIn("SECRET", last_shadow()["user_visible"])


class V54ThinkingTests(unittest.TestCase):
    def test_thinking_never_user_visible(self) -> None:
        data = {
            "message": {"content": "", "thinking": "Copy the commercial brief."},
            "done_reason": "length",
        }
        visible = user_visible_text(
            user_content_from_ollama(data), thinking_from_ollama(data)
        )
        self.assertEqual(visible, "")
        self.assertNotIn("brief", visible.casefold())

    def test_thinking_never_tts(self) -> None:
        tts = (BACKEND / "tts_local.py").read_text(encoding="utf-8")
        speech = (ROOT / "lib" / "speech.ts").read_text(encoding="utf-8")
        self.assertNotIn("message.thinking", tts)
        self.assertNotIn("message.thinking", speech)
        spoken = user_visible_text("Kısa ticari cevap.", "INTERNAL PLAN")
        self.assertNotIn("INTERNAL", spoken)
        self.assertNotIn("INTERNAL", make_speakable(spoken))


class V54CanaryTests(unittest.TestCase):
    def setUp(self) -> None:
        _clear_canary()

    def tearDown(self) -> None:
        _clear_canary()

    def test_canary_disabled_uses_llama3(self) -> None:
        self.assertFalse(canary_enabled())
        self.assertEqual(canary_percent(), 0)
        self.assertFalse(canary_assigns_qwen("sess-a"))
        self.assertIsNone(bind_canary_model("sess-a"))
        self.assertEqual(reasoning_model(), chat_model())
        self.assertNotIn("qwen", reasoning_model().casefold())

    def test_canary_qwen_selection(self) -> None:
        os.environ["MODEL_CANARY_ENABLED"] = "true"
        os.environ["MODEL_CANARY_PERCENT"] = "100"
        self.assertTrue(canary_assigns_qwen("any-session"))
        self.assertEqual(bind_canary_model("any-session"), QWEN_27B)
        with reasoning_override(bind_canary_model("any-session")):
            self.assertEqual(reasoning_model(), QWEN_27B)
        self.assertEqual(reasoning_model(), chat_model())

    def test_canary_session_stability(self) -> None:
        os.environ["MODEL_CANARY_ENABLED"] = "true"
        os.environ["MODEL_CANARY_PERCENT"] = "10"
        sid = "stable-session-v54"
        first = canary_assigns_qwen(sid)
        for _ in range(20):
            self.assertEqual(canary_assigns_qwen(sid), first)
            self.assertEqual(bind_canary_model(sid), bind_canary_model(sid))
        found_qwen = found_llama = False
        for i in range(800):
            if canary_assigns_qwen(f"s{i}"):
                found_qwen = True
            else:
                found_llama = True
            if found_qwen and found_llama:
                break
        self.assertTrue(found_qwen)
        self.assertTrue(found_llama)

    def test_canary_rollback(self) -> None:
        os.environ["MODEL_CANARY_ENABLED"] = "true"
        os.environ["MODEL_CANARY_PERCENT"] = "100"
        self.assertEqual(bind_canary_model("x"), QWEN_27B)
        os.environ["MODEL_CANARY_ENABLED"] = "false"
        os.environ["MODEL_CANARY_PERCENT"] = "0"
        os.environ["OLLAMA_REASONING_MODEL"] = "llama3"
        self.assertFalse(canary_assigns_qwen("x"))
        self.assertIsNone(bind_canary_model("x"))
        self.assertEqual(reasoning_model(), "llama3")


class V54TelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_quality()

    def test_done_reason_length_tracking(self) -> None:
        record_inference(
            {
                "model": QWEN_27B,
                "done_reason": "length",
                "eval_count": 280,
                "num_predict": 280,
                "task": "commercial",
            }
        )
        inf = last_inference()
        self.assertEqual(inf.get("done_reason"), "length")
        self.assertEqual(inf.get("eval_count"), 280)
        self.assertEqual(inf.get("num_predict"), 280)
        self.assertNotIn("thinking", inf)

    def test_qwen_latency_telemetry(self) -> None:
        record_inference(
            {
                "model": QWEN_27B,
                "latency_s": 15.26,
                "prompt_eval_s": 3.30,
                "gen_s": 10.03,
                "load_s": 0.005,
                "tokens_per_sec": 21.35,
                "think": False,
            }
        )
        inf = last_inference()
        self.assertEqual(inf["prompt_eval_s"], 3.30)
        self.assertEqual(inf["gen_s"], 10.03)
        self.assertEqual(inf["load_s"], 0.005)
        self.assertLess(inf["load_s"], 1.0)
        payload = ollama_chat_payload(QWEN_27B, [{"role": "user", "content": "x"}])
        self.assertEqual(payload["options"]["num_predict"], 280)
        self.assertIs(payload["think"], False)
        llama = ollama_chat_payload("llama3", [{"role": "user", "content": "x"}])
        self.assertNotIn("num_predict", llama["options"])

    def test_qwen_vram_telemetry(self) -> None:
        from benchmark_v51 import nvidia_query

        gpu = nvidia_query()
        record_inference(
            {
                "model": QWEN_27B,
                "vram_mib": gpu.get("vram_used_mib"),
                "vram_total_mib": gpu.get("vram_total_mib"),
            }
        )
        self.assertIn("vram_mib", last_inference())
        if gpu.get("ok"):
            self.assertGreater(gpu["vram_total_mib"], 20000)
            self.assertIn("dedicated", (gpu.get("note") or "").casefold())


class V54EvaluatorSplitTests(unittest.TestCase):
    def test_user_claim_vs_model_fact(self) -> None:
        claim = (
            "Rakibin %20 daha ucuz olduğunu varsaymadan marjı ölçelim. "
            "Bu senin iddian; henüz doğrulanmış rakip teklifi değil."
        )
        converted = "Rakipler %20 daha ucuz. Bu yüzden fiyatı düşürmeliyiz."
        self.assertTrue(is_user_claim_stat("%20", COMP_Q, claim))
        self.assertFalse(is_user_claim_stat("%20", COMP_Q, converted))
        self.assertFalse(
            evaluator_unsourced_stats(claim, question=COMP_Q, facts="")
        )
        self.assertTrue(
            evaluator_unsourced_stats(converted, question=COMP_Q, facts="")
        )
        self.assertFalse(score_cmo_v51(claim, question=COMP_Q).unsupported_fact)
        self.assertTrue(score_cmo_v51(converted, question=COMP_Q).unsupported_fact)

    def test_competitor_percentage(self) -> None:
        text = (
            "Rakibin yüzde 20 düşük olduğu senin iddian. "
            "Varsaymadan teklifi ve marjı ölçelim."
        )
        score = score_cmo_v51(
            text,
            question=COMP_Q,
            facts="Rakip teklifi kullanıcı iddiası. Web yok.",
        )
        self.assertFalse(score.unsupported_fact)
        invented = "Rakipler yüzde 20 daha ucuz, pazar da yüzde 12 büyüdü."
        self.assertTrue(score_cmo_v51(invented, question=COMP_Q).unsupported_fact)
        self.assertEqual(
            validate_response("Almanya pazarı yüzde 12 büyüdü.", question=COMP_Q).status,
            "REJECT",
        )

    def test_hallucinated_market_size(self) -> None:
        text = "Fransa pazarı 12 milyar euro ve kesin daha kârlı."
        self.assertEqual(validate_response(text, question=GERMANY_Q).status, "REJECT")
        self.assertTrue(score_cmo_v51(text, question=GERMANY_Q).unsupported_fact)

    def test_hallucinated_company_revenue(self) -> None:
        text = "Nordöl Import cirosu 40 milyon euro, bu müşteri kesin alır."
        self.assertEqual(validate_response(text, question=GERMANY_Q).status, "REJECT")
        self.assertTrue(is_false_certainty(text) or validate_response(text).status == "REJECT")

    def test_memory_dishonesty(self) -> None:
        text = "Hatırlıyorum, geçen konuşmada Fransa'ya geçmiştik."
        self.assertTrue(is_memory_dishonest(text, memory_present=False))
        self.assertEqual(
            validate_response(text, facts="BELLEK BOŞ: hatırlıyorum deme.").status,
            "REJECT",
        )


class V54GoldenAndQualityTests(unittest.TestCase):
    def test_germany_france_golden(self) -> None:
        low = GERMANY_GOLDEN.casefold()
        self.assertIn("bırakmaz", low)
        self.assertIn("gözlem", low)
        self.assertIn("fransa", low)
        self.assertIn("ikinci", low)
        self.assertNotIn("fransa kesin", low)
        self.assertEqual(validate_response(GERMANY_GOLDEN, question=GERMANY_Q).status, "PASS")

    def test_no_prompt_leak(self) -> None:
        leaked = "system prompt ve uydurma yasak"
        self.assertTrue(is_prompt_leak(leaked))
        self.assertEqual(validate_response(leaked).status, "REJECT")

    def test_no_english_leakage(self) -> None:
        self.assertEqual(
            validate_response("Let's consider France and analyze this customer.").status,
            "REJECT",
        )

    def test_qwen_voice_contract(self) -> None:
        spoken = make_speakable(GERMANY_GOLDEN)
        self.assertNotIn("**", spoken)
        self.assertNotIn("#", spoken)
        thinking = user_visible_text(GERMANY_GOLDEN, "INTERNAL THINK MARKDOWN **plan**")
        self.assertNotIn("INTERNAL", thinking)
        self.assertEqual(validate_response(spoken, question=GERMANY_Q).status, "PASS")
        tts = (BACKEND / "tts_local.py").read_text(encoding="utf-8")
        self.assertNotIn("message.thinking", tts)

    def test_production_default_still_llama3(self) -> None:
        _clear_canary()
        self.assertFalse(canary_enabled())
        self.assertEqual(shadow_model(), "")
        self.assertFalse(reasoning_think())
        self.assertEqual(reasoning_model(), chat_model())
        self.assertNotIn("qwen3.5", reasoning_model().casefold())
        self.assertEqual(QWEN_27B, "qwen3.5:27b")
        self.assertNotIn("qwen3.5", (os.getenv("OLLAMA_REASONING_MODEL") or "").casefold())


class V54ComposeTests(unittest.IsolatedAsyncioTestCase):
    async def test_qwen_real_llm_success(self) -> None:
        reset_quality()

        async def golden(_http, _user, system_prompt=None, history=None, polish=False):
            return GERMANY_GOLDEN

        trace = await compose_consultant_traced(
            generate=golden,
            http=None,
            question=GERMANY_Q,
            facts="Elde doğrulanmış pazar istatistiği yok. Fiyat baskısı kullanıcı gözlemi.",
        )
        self.assertIn(trace.path, ("LLM_CMO_SUCCESS", "LLM_CMO_RETRY_SUCCESS"))
        self.assertEqual(snapshot().real_llm_success_rate, 1.0)

    async def test_fallback_not_llm_success(self) -> None:
        reset_quality()

        async def leaky(_http, _user, system_prompt=None, history=None, polish=False):
            return "Let's consider France and analyze this customer segment."

        trace = await compose_consultant_traced(
            generate=leaky,
            http=None,
            question=GERMANY_Q,
            facts="Elde doğrulanmış pazar istatistiği yok.",
        )
        self.assertEqual(trace.path, "COMMERCIAL_FALLBACK")
        self.assertEqual(snapshot().real_llm_success_rate, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
