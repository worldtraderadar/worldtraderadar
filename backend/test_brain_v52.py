"""V5.2: Qwen3.5 27B benchmark regression. Eski 106 testi değiştirmez."""

from __future__ import annotations

import os
import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from benchmark_v51_cases import CASES
from llm import chat_model, reasoning_model


class V52FairnessTests(unittest.TestCase):
    def test_uses_same_twenty_v51_cases(self) -> None:
        self.assertEqual(len(CASES), 20)
        self.assertEqual(CASES[1]["id"], "germany_france")
        self.assertIn("Fransa'ya mı yönelmeliyim?", CASES[1]["input"])

    def test_production_reasoning_model_not_switched_to_qwen(self) -> None:
        self.assertNotIn("qwen3.5", (os.getenv("OLLAMA_REASONING_MODEL") or "").casefold())
        self.assertEqual(reasoning_model(), chat_model())

    def test_germany_case_still_second_in_pack(self) -> None:
        ids = [row["id"] for row in CASES]
        self.assertEqual(ids[1], "germany_france")
        self.assertIn("competitor_pressure", ids)
        self.assertIn("capacity", ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
