"""V5.2.1: thinking vs final content isolation. Eski 109 testi değiştirmez."""

from __future__ import annotations

import os
import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from benchmark_v51_cases import CASES
from benchmark_v521 import (
    MODEL,
    NUM_PREDICT_FAIR,
    VRAM_V52,
    chat_payload,
    classify_fields,
    classify_v52,
    thinking_from_ollama,
    user_content_from_ollama,
)
from llm import chat_model, reasoning_model


class V521ThinkingIsolationTests(unittest.TestCase):
    def test_uses_same_twenty_v51_cases(self) -> None:
        self.assertEqual(len(CASES), 20)
        self.assertEqual(CASES[1]["id"], "germany_france")
        self.assertIn("Fransa'ya mı yönelmeliyim?", CASES[1]["input"])

    def test_production_reasoning_model_not_switched_to_qwen(self) -> None:
        self.assertNotIn("qwen3.5", (os.getenv("OLLAMA_REASONING_MODEL") or "").casefold())
        self.assertEqual(reasoning_model(), chat_model())

    def test_fair_num_predict_stays_280(self) -> None:
        self.assertEqual(NUM_PREDICT_FAIR, 280)
        self.assertEqual(MODEL, "qwen3.5:27b")

    def test_payload_only_think_differs(self) -> None:
        on = chat_payload("qwen3.5:27b", "SYS", "USER", think=True)
        off = chat_payload("qwen3.5:27b", "SYS", "USER", think=False)
        self.assertEqual(on["options"], off["options"])
        self.assertEqual(on["messages"], off["messages"])
        self.assertEqual(on["options"]["num_predict"], 280)
        self.assertTrue(on["think"] is True)
        self.assertTrue(off["think"] is False)

    def test_thinking_is_never_used_as_user_content(self) -> None:
        data = {
            "message": {
                "role": "assistant",
                "content": "",
                "thinking": "I should copy the commercial brief and recommend France.",
            },
            "eval_count": 280,
            "done_reason": "length",
        }
        content = user_content_from_ollama(data)
        thinking = thinking_from_ollama(data)
        self.assertEqual(content, "")
        self.assertIn("commercial brief", thinking)
        flags = classify_fields(thinking, content, 280, 280)
        self.assertTrue(flags["FINAL_CONTENT_EMPTY"])
        self.assertTrue(flags["THINKING_PRESENT"])
        self.assertTrue(flags["TOKEN_BUDGET_CONSUMED_BY_THINKING"])
        self.assertFalse(flags["AUTO_REASONING_FAILURE"])

    def test_v52_zero_is_api_mode_when_think_false_has_content(self) -> None:
        label = classify_v52(
            {"content_empty_rate": 1.0, "thinking_present_rate": 1.0},
            {
                "final_content_rate": 1.0,
                "summary": {"real_llm_success_rate": 0.4},
            },
        )
        self.assertEqual(label, "V5.2_WAS_API_THINKING_MODE_NOT_MODEL_REASONING_FAILURE")

    def test_vram_not_rerun_uses_v52_reference(self) -> None:
        self.assertEqual(VRAM_V52["MODEL_ONLY_PEAK_VRAM"], 17948)
        self.assertEqual(VRAM_V52["XTTS_MODEL_PEAK_VRAM"], 19925)


if __name__ == "__main__":
    unittest.main(verbosity=2)
