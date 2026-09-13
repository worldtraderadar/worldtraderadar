"""Commercial Intelligence / CMO davranış testleri.

Her senaryo şunu basar: INPUT / EXPECTED BEHAVIOR / ACTUAL / PASS/FAIL
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.commercial import (
    build_commercial_brief,
    commercial_fallback_reply,
    extract_user_claims,
    is_weak_decision,
    prioritize_matches,
    treats_score_as_win,
)
from agents.contracts import current_data_facts, memory_miss_facts, tool_failed_facts
from agents.goals import classify_goal, plan_summary
from agents.intent import classify_intent
from agents.response_engine import compose_consultant_reply, safe_consultant_fallback
from agents.session import SessionState
from agents.tools_base import ToolResult
from agents.validator import english_leak_score, make_speakable, validate_response
from prompts import is_decision_question, is_diagnostic_request


def _cards() -> list[dict]:
    return [
        {
            "organization_name": "Nordöl Import",
            "destination_country": "DE",
            "similarity": 0.88,
            "website": "https://example.de",
        },
        {
            "organization_name": "Boutique Huile",
            "destination_country": "FR",
            "similarity": 0.71,
            "has_contact": False,
        },
        {
            "organization_name": "Retail Mix GmbH",
            "destination_country": "DE",
            "similarity": 0.41,
        },
        {
            "organization_name": "Generic Shop",
            "destination_country": "NL",
            "similarity": 0.22,
        },
        {
            "organization_name": "Gastro Nord",
            "destination_country": "DE",
            "similarity": 0.79,
            "has_contact": True,
        },
        {
            "organization_name": "Huile Sud",
            "destination_country": "FR",
            "similarity": 0.55,
        },
        {
            "organization_name": "Market Wide",
            "destination_country": "DE",
            "similarity": 0.33,
        },
    ]


class ScenarioMixin:
    def report(self, inp: str, expected: str, actual: str, ok: bool) -> None:
        status = "PASS" if ok else "FAIL"
        block = (
            f"\n========== {self._testMethodName} ==========\n"
            f"INPUT: {inp}\n"
            f"EXPECTED BEHAVIOR: {expected}\n"
            f"ACTUAL: {actual}\n"
            f"{status}\n"
        )
        print(block, flush=True)
        self.assertTrue(ok, block)


class CommercialIntelligenceTests(ScenarioMixin, unittest.TestCase):
    def test_01_germany_or_france_keeps_existing_leads(self) -> None:
        q = (
            "Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. "
            "Fransa'ya mı yönelmeliyim?"
        )
        brief = build_commercial_brief(q)
        reply = commercial_fallback_reply(brief, q) or ""
        low = reply.casefold()
        ok = (
            is_decision_question(q)
            and classify_intent(q)[0] == "trade_advisor"
            and "3" in reply
            and "bırakmaz" in low
            and "fransa" in low
            and "analiz edelim" not in low
            and not is_weak_decision(reply)
        )
        self.report(
            q,
            "Almanya'yı hemen terk etme; 3 müşteriyi hesaba kat; Fransa ikinci test; zayıf kaçış yok",
            reply,
            ok,
        )

    def test_02_should_i_cut_price(self) -> None:
        q = "Fiyatı düşürmeli miyim?"
        brief = build_commercial_brief(q)
        reply = commercial_fallback_reply(brief, q) or ""
        low = reply.casefold()
        ok = (
            brief.mode == "decision"
            and "düşürmez" in low
            and "marj" in low
            and "teklif" in low
        )
        self.report(q, "Net duruş: fiyatı hemen düşürme; marjı ölç; sonraki adım ver", reply, ok)

    def test_03_sales_dropped_diagnoses_first(self) -> None:
        q = "Satışlarımız düştü, neden?"
        brief = build_commercial_brief(q)
        reply = commercial_fallback_reply(brief, q) or ""
        ok = (
            is_diagnostic_request(q)
            and brief.mode == "diagnostic"
            and "mevcut" in reply.casefold()
            and "yeni" in reply.casefold()
            and reply.count("?") <= 3
        )
        self.report(
            q,
            "Hemen çözüm dayatma; teşhis sorusu (mevcut vs yeni müşteri); en fazla 3 soru",
            reply,
            ok,
        )

    def test_04_new_vs_existing_customer_question(self) -> None:
        q = "Yeni müşteri mi mevcut müşteri mi?"
        brief = build_commercial_brief(q)
        reply = commercial_fallback_reply(brief, q) or ""
        ok = brief.mode == "diagnostic" and "mevcut" in reply.casefold() and "yeni" in reply.casefold()
        self.report(q, "Teşhis modu: düşüş kaynağını mevcut vs yeni diye ayır", reply, ok)

    def test_05_best_buyer_does_not_invent(self) -> None:
        q = "En iyi buyer hangisi?"
        brief = build_commercial_brief(q, matches=_cards())
        reply = commercial_fallback_reply(brief, q) or ""
        ok = (
            brief.mode == "buyer_priority"
            and "Nordöl" in reply
            and not treats_score_as_win(reply)
            and "uydurma" not in reply.casefold()
        )
        self.report(q, "Karttaki yüksek eşleşmeyi A önceliği yap; firma uydurma", reply, ok)

    def test_06_seven_buyers_abc_priority(self) -> None:
        q = "Bu 7 alıcıdan hangilerine öncelik vermeliyim?"
        ranked = prioritize_matches(_cards())
        brief = build_commercial_brief(q, matches=_cards())
        reply = commercial_fallback_reply(brief, q) or ""
        bands = {item.name: item.band for item in ranked}
        ok = (
            len(ranked) == 7
            and bands["Nordöl Import"] == "A"
            and bands["Generic Shop"] == "C"
            and "Hemen temas" in reply
            and "Araştır" in reply
        )
        self.report(
            q,
            "7 kaydı A/B/C'ye çevir; yüksek skor+pazar=A, yüzeysel=C; gerekçe kısa",
            f"{reply} | bands={bands}",
            ok,
        )

    def test_07_matching_score_is_not_win_probability(self) -> None:
        q = "%92 skor var, bu müşteri olur mu?"
        brief = build_commercial_brief(q)
        reply = commercial_fallback_reply(brief, q) or "Yüksek eşleşme skoru ticari uygunluk değildir."
        bad = "%92 ihtimalle müşteri olur"
        ok = (
            brief.mode == "matching"
            and not treats_score_as_win(reply)
            and treats_score_as_win(bad)
            and validate_response(bad, question=q).status == "REJECT"
            and "uygunluğ" in reply.casefold()
        )
        self.report(q, "Eslestirme skoru kazanma ihtimali degil; %92 musteri olur REJECT", reply, ok)

    def test_08_user_claim_is_not_objective_fact(self) -> None:
        q = "Almanya kötü, Fransa'ya geçelim. Fiyat baskısı çok yüksek."
        claims = extract_user_claims(q)
        brief = build_commercial_brief(q)
        prompt = brief.as_prompt()
        ok = (
            any("fiyat" in c.casefold() for c in claims)
            and "objective fact" in prompt.casefold()
            and "kullanıcı" in prompt.casefold()
        )
        self.report(q, "Fiyat baskisi / Almanya kotu kullanici iddiasi; OBJECTIVE FACT degil", prompt, ok)

    def test_09_asks_one_critical_question_when_data_missing(self) -> None:
        q = (
            "Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. "
            "Fransa'ya mı yönelmeliyim?"
        )
        brief = build_commercial_brief(q)
        ok = 1 <= len(brief.critical_questions) <= 3 and "marj" in brief.critical_questions[0].casefold()
        self.report(
            q,
            "Kritik veri eksikse 1–3 kısa ticari soru (marj / hacim)",
            str(brief.critical_questions),
            ok,
        )

    def test_10_net_decision_when_leads_exist(self) -> None:
        q = (
            "Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. "
            "Fransa'ya mı yönelmeliyim?"
        )
        reply = commercial_fallback_reply(build_commercial_brief(q), q) or ""
        ok = "bırakmaz" in reply.casefold() and "ikisi de" not in reply.casefold()
        self.report(q, "Yeterli sinyal varsa net öneri: Almanya'yı bırakma", reply, ok)

    def test_11_blocks_both_are_fine_escape(self) -> None:
        weak = "İkisi de değerlendirilebilir. Fransa'yı da analiz edelim."
        ok = is_weak_decision(weak) and validate_response(weak).status in ("WARN", "REJECT")
        self.report(
            weak,
            "«İkisi de olabilir / Fransa'yı da analiz» zayıf kaçış sayılır",
            f"weak={is_weak_decision(weak)} verdict={validate_response(weak).status}",
            ok,
        )

    def test_12_uses_memory_when_present(self) -> None:
        q = "Almanya mı Fransa mı?"
        session = SessionState(
            session_id="mem-hit",
            memories=["Geçen tur Almanya'da distribütör modeliyle ilerleme kararı aldık."],
        )
        brief = build_commercial_brief(q, session=session)
        ok = "distribütör" in brief.as_prompt().casefold() and "BELLEK BOŞ" not in brief.as_prompt()
        self.report(q, "Memory varsa brife girer ve karara katılır", brief.as_prompt(), ok)

    def test_13_does_not_invent_memory(self) -> None:
        q = "Geçen hafta neye karar vermiştik?"
        session = SessionState(session_id="mem-miss")
        brief = build_commercial_brief(q, session=session)
        missed = safe_consultant_fallback(
            question=q,
            facts="",
            tools=[
                ToolResult(
                    name="memory_search",
                    ok=False,
                    summary="yok",
                    provenance="TOOL_FAILED",
                )
            ],
            session=session,
        )
        ok = (
            "BELLEK BOŞ" in brief.as_prompt()
            and missed == memory_miss_facts()
            and "hatırlıyorum" not in missed.casefold()
        )
        self.report(q, "Memory yoksa hatırlıyorum deme; dürüst boş bellek", missed, ok)

    def test_14_company_profile_enters_decision(self) -> None:
        q = "Almanya mı Fransa mı?"
        session = SessionState(
            session_id="co",
            product="zeytinyağı",
            capacity="20 ton / ay",
            market="Almanya",
        )
        brief = build_commercial_brief(q, session=session)
        prompt = brief.as_prompt().casefold()
        reply = commercial_fallback_reply(brief, q) or brief.stance
        ok = (
            "zeytinyağı" in prompt
            and "20 ton" in prompt
            and "almanya" in prompt
            and "zeytinyağı" in (reply + " " + brief.stance).casefold()
        )
        self.report(q, "Şirket profili (ürün/kapasite/pazar) karara girer", brief.as_prompt() + " | " + reply, ok)

    def test_15_web_hit_allows_only_sourced_stats(self) -> None:
        q = "Bugün Almanya zeytinyağı piyasasında ne oldu, Fransa'ya geçeyim mi?"
        facts = "Kaynak: ithalat yüzde 4 arttı."
        tools = [
            ToolResult(
                name="web_search",
                ok=True,
                summary=facts,
                provenance="SOURCE_WEB",
            )
        ]
        good = validate_response(
            "Mevcut verilere göre ithalat yüzde 4 arttı. Yine de Almanya'yı bırakmazdım.",
            question=q,
            facts=facts,
            tools=tools,
        )
        bad = validate_response(
            "Pazar yüzde 31 büyüdü, Fransa kesin daha kârlı.",
            question=q,
            facts=facts,
            tools=tools,
        )
        ok = good.status == "PASS" and bad.status == "REJECT"
        self.report(
            q,
            "Web varsa yalnızca kaynaklı rakam; uydurma yüzde REJECT",
            f"sourced={good.status} invented={bad.status}",
            ok,
        )

    def test_16_no_web_does_not_invent_current_numbers(self) -> None:
        q = "Almanya'da zeytinyağı pazarı 2026'da yüzde kaç büyüdü?"
        safe = safe_consultant_fallback(
            question=q,
            facts=current_data_facts(),
            tools=[
                ToolResult(
                    name="web_search",
                    ok=False,
                    summary="yok",
                    provenance="TOOL_FAILED",
                )
            ],
        )
        invented = validate_response(
            "2026'da pazar yüzde 12 büyüdü.",
            question=q,
            facts=current_data_facts(),
            tools=[
                ToolResult(
                    name="web_search",
                    ok=False,
                    summary="yok",
                    provenance="TOOL_FAILED",
                )
            ],
        )
        ok = invented.status == "REJECT" and not any(
            token in safe for token in ("%", "yüzde 12", "milyon")
        )
        self.report(q, "Web yoksa güncel rakam uydurma", f"fallback={safe} invented={invented.status}", ok)

    def test_17_tool_failure_does_not_claim_found(self) -> None:
        q = "Almanya'da müşteri bul."
        failed = ToolResult(
            name="buyer_search",
            ok=False,
            summary="yok",
            provenance="TOOL_FAILED",
        )
        safe = safe_consultant_fallback(question=q, facts="", tools=[failed])
        verdict = validate_response(
            "37 firma buldum.",
            question=q,
            tools=[failed],
        )
        ok = (
            safe == tool_failed_facts("Firma araması")
            and verdict.status == "REJECT"
            and "37" not in safe
        )
        self.report(q, "Tool failure sonrası «buldum» / sayı yok", f"{safe} | {verdict.status}", ok)

    def test_18_turkish_commercial_language(self) -> None:
        q = (
            "Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. "
            "Fransa'ya mı yönelmeliyim?"
        )
        reply = commercial_fallback_reply(build_commercial_brief(q), q) or ""
        leak, why = english_leak_score(reply)
        ok = not leak and validate_response(reply, question=q).status == "PASS"
        self.report(q, "Doğal Türkçe ticari dil; İngilizce sızıntı yok", f"{reply} | leak={why}", ok)

    def test_19_mixed_english_leakage_rejected(self) -> None:
        samples = [
            "Fransa'yı da consider edelim.",
            "Rakibi analyze edelim.",
            "Bu customer segment daha iyi.",
        ]
        results = [(s, validate_response(s).status, english_leak_score(s)[0]) for s in samples]
        allowed = validate_response("B2B, MOQ ve FOB netleşmeden CIF kilitlemezdim.").status
        ok = all(status == "REJECT" and leak for _, status, leak in results) and allowed == "PASS"
        self.report(
            " | ".join(samples),
            "consider/analyze/customer REJECT; B2B/MOQ/FOB serbest",
            str(results) + f" allowed={allowed}",
            ok,
        )

    def test_20_speakable_natural_voice_format(self) -> None:
        q = (
            "Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. "
            "Fransa'ya mı yönelmeliyim?"
        )
        reply = commercial_fallback_reply(build_commercial_brief(q), q) or ""
        spoken = make_speakable(reply)
        ok = (
            "http" not in spoken.casefold()
            and "##" not in spoken
            and "|" not in spoken
            and spoken.count("\n") <= 4
            and "ben şu aşamada" in spoken.casefold()
        )
        self.report(q, "Sesli okunabilir: tablo/markdown/URL yok, doğal danışman cümlesi", spoken, ok)

    def test_21_human_plan_hides_internal_machinery(self) -> None:
        q = (
            "Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. "
            "Fransa'ya mı yönelmeliyim?"
        )
        plan = plan_summary(classify_goal(q), q, None)
        blob = plan.casefold()
        ok = (
            "llm" not in blob
            and "validator" not in blob
            and "prompt" not in blob
            and "tool registry" not in blob
            and "fiyat" in blob
        )
        self.report(q, "Plan UI insan dilinde; iç teknik yapı yok", plan, ok)

    def test_22_user_claim_price_pressure_framed_in_reply(self) -> None:
        q = (
            "Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. "
            "Fransa'ya mı yönelmeliyim?"
        )
        reply = commercial_fallback_reply(build_commercial_brief(q), q) or ""
        low = reply.casefold()
        ok = "gözlem" in low or "senin" in low
        ok = ok and "almanya'da fiyat baskısı yüksek" not in low
        self.report(
            q,
            "Fiyat baskısını kullanıcı iddiası olarak çerçevele; objektif gerçek yapma",
            reply,
            ok,
        )

    def test_25_mail_draft_does_not_leak_into_decision(self) -> None:
        q = (
            "Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. "
            "Fransa'ya mı yönelmeliyim?"
        )
        mail = (
            "Dear Spielwarenmesse,\n\n%30 peşin, %70 yüklemede.\n\n"
            "Best regards\n[Your Company Name]"
        )
        verdict = validate_response(mail, question=q)
        ok = verdict.status == "REJECT" and "mail" in " ".join(verdict.reasons).casefold()
        self.report(q, "Karar sorusunda mail taslagi / Dear / Best regards REJECT", str(verdict), ok)

    def test_26_false_confidence_is_rejected(self) -> None:
        bad = "Fransa kesin daha kârlı. Bu müşteri kesin alır."
        good = "Mevcut sinyallere göre Almanya'yı bırakmazdım. Kararı değiştirecek veri marj."
        v_bad = validate_response(bad)
        v_good = validate_response(good)
        ok = v_bad.status == "REJECT" and v_good.status == "PASS"
        self.report(
            bad,
            "kesin daha karli / kesin alir REJECT; mevcut sinyallere gore PASS",
            f"bad={v_bad.status} good={v_good.status}",
            ok,
        )

    def test_27_intent_ui_is_human_not_scores(self) -> None:
        q = (
            "Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. "
            "Fransa'ya mı yönelmeliyim?"
        )
        from agents.commercial import human_intent_detail

        line = human_intent_detail("trade_advisor", q)
        blob = line.casefold()
        ok = (
            "chat 0" not in blob
            and "advisor 1" not in blob
            and "llm" not in blob
            and "validator" not in blob
            and "ajan" not in blob
            and "karar" in blob
        )
        self.report(q, "Niyet satiri insan dilinde; skor / ajan / LLM yok", line, ok)


class CommercialComposeTests(ScenarioMixin, unittest.IsolatedAsyncioTestCase):
    async def test_23_leaky_llm_falls_back_to_cmo(self) -> None:
        q = (
            "Almanya'dan 3 müşteri buldum ama fiyat baskısı çok yüksek. "
            "Fransa'ya mı yönelmeliyim?"
        )

        async def leaky(_http, _user, system_prompt=None, history=None, polish=False):
            return "Let's consider France and analyze this customer segment."

        actual = await compose_consultant_reply(
            generate=leaky,
            http=None,
            question=q,
            facts="Elde doğrulanmış pazar istatistiği yok.",
        )
        low = actual.casefold()
        ok = (
            "consider" not in low
            and "customer" not in low
            and "bırakmaz" in low
            and "3" in actual
        )
        self.report(
            q,
            "LLM İngilizce/leakage üretirse CMO fallback: Almanya'yı bırakma + 3 müşteri",
            actual,
            ok,
        )


class VoiceRegressionGuardTests(ScenarioMixin, unittest.TestCase):
    def test_24_voice_stack_untouched(self) -> None:
        root = Path(__file__).resolve().parents[1]
        files = [
            root / "backend" / "tts_local.py",
            root / "lib" / "speech.ts",
            root / "components" / "mic-button.tsx",
            root / "components" / "speak-button.tsx",
            root / "app" / "api" / "speak" / "route.ts",
        ]
        missing = [str(path) for path in files if not path.exists()]
        text = (root / "backend" / "tts_local.py").read_text(encoding="utf-8")
        ok = not missing and "xtts" in text.casefold() and "piper" in text.casefold()
        self.report(
            "voice stack files",
            "XTTS / Piper / mic / speak API dosyaları yerinde ve bu turda dokunulmamış",
            f"missing={missing} xtts={'xtts' in text.casefold()}",
            ok,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
