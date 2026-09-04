"""Keşif oturumu: ürün sonrası seçenek; ardışık soru yok."""

from __future__ import annotations

import unittest

from agents.intent import classify_intent
from agents.session import (
    SessionState,
    continue_intake,
    director_options_reply,
    hydrate,
)
from prompts import GREETING_REPLY, INTAKE_REPLY, is_small_talk


class SessionIntentTests(unittest.TestCase):
    def test_short_product_is_not_small_talk(self) -> None:
        self.assertFalse(is_small_talk("dokuma etiket"))
        self.assertFalse(is_small_talk("dokuma etiket", has_history=True))

    def test_greeting_without_history_is_chat(self) -> None:
        self.assertTrue(is_small_talk("merhaba, nasılsınız?"))
        intent, _ = classify_intent("merhaba, nasılsınız?")
        self.assertEqual(intent, "chat")

    def test_product_followup_offers_options_not_capacity_ask(self) -> None:
        session = SessionState(session_id="t1", last_ask="product")
        session.append("user", "Satış yapmak istiyorum")
        session.append("assistant", INTAKE_REPLY)
        intent, _ = classify_intent("dokuma etiket", session=session)
        self.assertEqual(intent, "intake")
        reply = continue_intake(session, "dokuma etiket")
        self.assertIn("dokuma etiket", reply.casefold())
        self.assertNotIn("kapasiteniz", reply.casefold())
        self.assertNotIn("yurtiçi", reply.casefold())
        self.assertNotIn("hitap", reply.casefold())
        self.assertNotIn("hangisiyle", reply.casefold())
        self.assertNotIn("nasıl ilerleyelim", reply.casefold())
        self.assertNotIn("?", reply)
        self.assertIn("eşleştiniz", reply.casefold())
        self.assertIn("mail taslağı", reply.casefold())
        self.assertIn("teklif atın", reply.casefold())
        self.assertNotIn("isterseniz", reply.casefold())
        self.assertNotIn("subject:", reply.casefold())
        self.assertNotEqual(reply, GREETING_REPLY)
        self.assertEqual(session.product, "dokuma etiket")
        self.assertIsNone(session.last_ask)

    def test_history_hydrate_recovers_product_slot(self) -> None:
        history = [
            {"role": "user", "content": "Satış yapamıyorum"},
            {"role": "assistant", "content": INTAKE_REPLY},
        ]
        state = hydrate("abc", history)
        self.assertEqual(state.last_ask, "product")
        intent, _ = classify_intent("dokuma etiket", session=state)
        self.assertEqual(intent, "intake")

    def test_capacity_triggers_director_options(self) -> None:
        session = SessionState(
            session_id="t2", product="dokuma etiket", last_ask="choice"
        )
        reply = continue_intake(session, "2.5 milyon mt")
        self.assertIn("2.5 milyon metre", reply)
        self.assertIn("dokuma etiket", reply.casefold())
        self.assertNotIn("hangisiyle", reply.casefold())
        self.assertNotIn("nasıl ilerleyelim", reply.casefold())
        self.assertNotIn("yurtiçi", reply.casefold())
        self.assertNotIn("kapasiteniz nedir", reply.casefold())
        self.assertNotIn("?", reply)
        self.assertIsNone(session.last_ask)
        self.assertEqual(session.capacity, "Aylık 2.5 milyon metre")
        self.assertTrue(reply.startswith("Sizin Aylık 2.5 milyon metre"))
        self.assertIn("dokuma etiket ürününüze", reply.casefold())
        self.assertIn("eşleştiniz", reply.casefold())
        self.assertIn("teklif atın", reply.casefold())
        self.assertNotIn("isterseniz", reply.casefold())
        self.assertNotIn("subject:", reply.casefold())

    def test_capacity_followup_routes_to_intake_not_reach(self) -> None:
        from agents.session import advisor_answer

        session = SessionState(
            session_id="cap-route",
            product="dokuma etiket",
            last_ask="choice",
            last_advisor_text=(
                "Sizin dokuma etiket ürününüze uygun olarak pazarınızdaki "
                "şu firmalarla eşleştiniz: Almanya alıcısı · Woven clothing labels. "
                "İsterseniz görüşmeniz için hemen İngilizce/Almanca özel bir teklif "
                "ve mail taslağı hazırlayabilirim."
            ),
            last_advisor_kind="mediate",
        )
        intent, _ = classify_intent("2.5 milyon mt", session=session)
        self.assertEqual(intent, "intake")
        reply = continue_intake(session, "2.5 milyon mt")
        self.assertEqual(session.capacity, "Aylık 2.5 milyon metre")
        self.assertIn("kapasitenize", reply.casefold())
        self.assertNotIn("nasıl ulaşacağınız", reply.casefold())
        noted = advisor_answer("2.5 milyon mt", session, matches=[])
        self.assertIn("Sizin Aylık 2.5 milyon metre kapasitenize", noted)
        self.assertNotIn("nasıl ulaşacağınız", noted.casefold())
        self.assertIn("teklif atın", noted.casefold())

    def test_option_pick_routes_to_supplier(self) -> None:
        session = SessionState(
            session_id="t3", product="dokuma etiket", last_ask="choice"
        )
        intent, _ = classify_intent("Avrupa'daki üreticilere bağlayalım", session=session)
        self.assertEqual(intent, "buyer_finder")

    def test_buyer_ask_bypasses_intake_questions(self) -> None:
        session = SessionState(
            session_id="t4", product="dokuma etiket", last_ask="choice"
        )
        intent, _ = classify_intent("alıcıları bulabilir misin?", session=session)
        self.assertEqual(intent, "buyer_finder")
        self.assertIsNone(session.last_ask)

    def test_musteri_ariyorum_is_buyer_finder(self) -> None:
        session = SessionState(session_id="t5", product="dokuma etiket")
        intent, _ = classify_intent("müşteri arıyorum", session=session)
        self.assertEqual(intent, "buyer_finder")

    def test_buyer_action_reply_has_no_question(self) -> None:
        from agents.session import party_action_reply

        reply = party_action_reply(
            role="buyer", product="dokuma etiket", matches=[]
        )
        self.assertEqual(
            reply,
            "Hemen filtreliyorum; Dokuma etiket için Almanya ve İtalya'daki "
            "potansiyel alıcı listesini çıkarıyorum.",
        )
        self.assertNotIn("?", reply)

        raw = "Zeytinyağı için alıcıları bulabilir misin?"
        olive = party_action_reply(role="buyer", product=raw, question=raw)
        self.assertEqual(
            olive,
            "Hemen filtreliyorum; Zeytinyağı için Almanya ve İtalya'daki "
            "potansiyel alıcı listesini çıkarıyorum.",
        )
        self.assertNotIn("bulabilir", olive.casefold())
        self.assertFalse(olive.casefold().startswith(raw.casefold()))

        supplier = party_action_reply(role="supplier", product="zeytinyağı")
        self.assertEqual(
            supplier,
            "Hemen filtreliyorum; Zeytinyağı için Almanya ve İtalya'daki "
            "potansiyel tedarikçi listesini çıkarıyorum.",
        )

    def test_director_template_matches_brief(self) -> None:
        reply = director_options_reply("dokuma etiket", "2.5 milyon mt")
        self.assertTrue(reply.startswith("Sizin Aylık 2.5 milyon metre"))
        self.assertIn("dokuma etiket ürününüze", reply.casefold())
        self.assertIn("eşleştiniz", reply.casefold())
        self.assertIn("İngilizce", reply)
        self.assertIn("Almanca", reply)
        self.assertIn("mail taslağı", reply.casefold())
        self.assertIn("teklif atın", reply.casefold())
        self.assertNotIn("isterseniz", reply.casefold())
        self.assertNotIn("hangisiyle", reply.casefold())
        self.assertNotIn("nasıl ilerleyelim", reply.casefold())
        self.assertNotIn("?", reply)
        self.assertNotIn("subject:", reply.casefold())

    def test_dotted_millions_are_not_truncated(self) -> None:
        from agents.session import (
            capacity_reminder,
            extract_capacity,
            format_capacity,
            session_notes,
            split_product_and_capacity,
        )

        for raw in ("2.500.000 mt", "2.500.000 metre", "aylık 2.500.000 mt"):
            canonical = extract_capacity(raw)
            self.assertEqual(canonical, "Aylık 2.5 milyon metre")
            self.assertNotIn("2.500 ", f"{canonical} ")

        product, cap = split_product_and_capacity("dokuma etiket 2.500.000 mt")
        self.assertEqual(product, "dokuma etiket")
        self.assertEqual(cap, "Aylık 2.5 milyon metre")

        session = SessionState(
            session_id="cap-dots", product="dokuma etiket", last_ask="choice"
        )
        reply = continue_intake(session, "2.500.000 mt")
        self.assertEqual(session.capacity, "Aylık 2.5 milyon metre")
        self.assertTrue(reply.startswith("Sizin Aylık 2.5 milyon metre"))
        self.assertEqual(
            capacity_reminder("dokuma etiket", "2.500.000 mt"),
            "Aylık 2.5 milyon metre dokuma etiket",
        )
        self.assertEqual(format_capacity("2.5 milyon mt"), "Aylık 2.5 milyon metre")

        notes = session_notes(session)
        self.assertIn("basamak kırpma yasak", notes)
        self.assertIn("Aylık 2.5 milyon metre", notes)
        self.assertIn("2.5 milyon metre dokuma etiket", notes.casefold())

    def test_hydrate_keeps_full_capacity_scale(self) -> None:
        from agents.session import session_notes

        history = [
            {"role": "user", "content": "dokuma etiket"},
            {
                "role": "assistant",
                "content": director_options_reply("dokuma etiket"),
            },
            {"role": "user", "content": "2.500.000 mt"},
        ]
        state = hydrate("cap-hist", history)
        self.assertEqual(state.product, "dokuma etiket")
        self.assertEqual(state.capacity, "Aylık 2.5 milyon metre")
        notes = session_notes(state)
        self.assertIn("2.5 milyon", notes)
        self.assertIn("dokuma etiket", notes)

    def test_strategy_request_routes_to_advisor_without_question(self) -> None:
        from agents.session import advisor_report_reply, report_is_complete
        from prompts import is_sector_chat, is_strategy_request, polish_chat_reply

        session = SessionState(
            session_id="adv1",
            product="dokuma etiket",
            capacity="Aylık 2.5 milyon metre",
            last_ask="choice",
        )
        for phrase in ("strateji hazarla", "analiz yap", "rapor ver"):
            session.last_ask = "choice"
            self.assertTrue(is_strategy_request(phrase), phrase)
            self.assertFalse(is_sector_chat(phrase))
            intent, _ = classify_intent(phrase, session=session)
            self.assertEqual(intent, "trade_advisor", phrase)
            self.assertIsNone(session.last_ask)

        report = advisor_report_reply(
            "dokuma etiket", "Aylık 2.5 milyon metre", matches=[]
        )
        self.assertTrue(report_is_complete(report))
        self.assertNotIn("?", report)
        self.assertNotIn("hangisiyle", report.casefold())
        self.assertIn("Pazar giriş adımları", report)
        self.assertIn("Hedef müşteri profilleri", report)
        self.assertIn("Lojistik ve gümrük ipuçları", report)
        self.assertIn("2.5 milyon metre", report)
        self.assertIn("dokuma etiket", report.casefold())
        self.assertIn("teklif atın", report.casefold())
        self.assertIn("0,028", report)
        banned = (
            "renk haslığı",
            "damask",
            "woven",
            "atr",
            "eur.1",
            "private-label",
            "hs 5807",
            "tercihî",
            "tercihli tarife",
        )
        folded = report.casefold()
        for term in banned:
            self.assertNotIn(term, folded, term)

        generic = advisor_report_reply("mobilya", None, matches=[])
        self.assertTrue(report_is_complete(generic))
        generic_fold = generic.casefold()
        for term in banned:
            self.assertNotIn(term, generic_fold, term)

        leaked = polish_chat_reply(
            report + " Hangisiyle başlayalım?",
            advisor_report=True,
        )
        self.assertNotIn("hangisiyle", leaked.casefold())
        self.assertNotIn("?", leaked)
        self.assertIn("Pazar giriş adımları", leaked)

    def test_method_question_does_not_repeat_template_report(self) -> None:
        from agents.session import advisor_answer
        from prompts import advisor_focus, is_method_question

        session = SessionState(
            session_id="adv-method",
            product="dokuma etiket",
            capacity="Aylık 2.5 milyon metre",
            last_ask="choice",
        )
        for phrase in ("nasıl ulaşacağım?", "hangi fuara gideyim?"):
            session.last_ask = "choice"
            self.assertTrue(is_method_question(phrase), phrase)
            intent, _ = classify_intent(phrase, session=session)
            self.assertEqual(intent, "trade_advisor", phrase)
            self.assertIsNone(session.last_ask)

        self.assertEqual(advisor_focus("hangi fuara gideyim?"), "fair")
        self.assertEqual(advisor_focus("linkedin ile nasıl ulaşırım"), "linkedin")
        self.assertEqual(advisor_focus("nasıl ulaşacağım?"), "reach")
        self.assertEqual(advisor_focus("strateji hazarla"), "report")
        self.assertEqual(advisor_focus("yabancı dilim yok"), "language")
        self.assertEqual(advisor_focus("e-posta şablonu ver"), "draft")
        self.assertEqual(advisor_focus("bana mail taslağı hazırlarsın"), "draft")

        reach = advisor_answer("nasıl ulaşacağım?", session)
        self.assertNotIn("Pazar giriş adımları", reach)
        self.assertNotIn("Hedef müşteri profilleri", reach)
        self.assertNotIn("?", reach)
        self.assertIn("linkedin", reach.casefold())
        self.assertTrue(
            "texworld" in reach.casefold() or "eurocetex" in reach.casefold()
        )

        fair = advisor_answer("hangi fuara gideyim?", session)
        self.assertNotIn("Pazar giriş adımları", fair)
        self.assertIn("texworld", fair.casefold())
        self.assertIn("eurocetex", fair.casefold())
        self.assertNotEqual(fair.strip(), reach.strip())

        first = advisor_answer("strateji hazarla", SessionState(
            session_id="dup1", product="dokuma etiket", capacity="Aylık 2.5 milyon metre"
        ))
        session_dup = SessionState(
            session_id="dup2",
            product="dokuma etiket",
            capacity="Aylık 2.5 milyon metre",
        )
        a = advisor_answer("strateji hazarla", session_dup)
        b = advisor_answer("strateji hazarla", session_dup)
        self.assertIn("Pazar giriş adımları", a)
        self.assertNotEqual(a.strip(), b.strip())
        self.assertNotIn("Pazar giriş adımları", b)
        self.assertTrue("fuar" in b.casefold() or "linkedin" in b.casefold())
        self.assertIn("Pazar giriş adımları", first)

    def test_language_barrier_returns_copy_paste_templates(self) -> None:
        from agents.session import advisor_answer
        from prompts import is_language_barrier

        session = SessionState(
            session_id="lang1",
            product="dokuma etiket",
            capacity="Aylık 2.5 milyon metre",
        )
        self.assertTrue(is_language_barrier("yabancı dilim yok"))
        self.assertTrue(is_language_barrier("İngilizcem yok"))
        intent, _ = classify_intent("yabancı dilim yok", session=session)
        self.assertEqual(intent, "trade_advisor")
        reply = advisor_answer("yabancı dilim yok", session)
        folded = reply.casefold()
        self.assertNotIn("hangisiyle", folded)
        self.assertNotIn("nasıl ilerleyelim", folded)
        self.assertNotIn("?", reply)
        self.assertIn("kopyalayıp", folded)
        self.assertIn("subject:", folded)
        self.assertIn("dear purchasing", folded)
        self.assertIn("betreff:", folded)
        self.assertIn("Google Translate", reply)
        self.assertIn("thank you. sample is ready", folded)
        self.assertIn("numune", folded)

    def test_mediate_names_matching_firms(self) -> None:
        from agents.session import advisor_mediate_reply

        matches = [
            {
                "organization_name": "NordTextil GmbH",
                "product_name": "Woven clothing labels (dokuma etiket)",
                "destination_country": "DE",
            },
            {
                "organization_name": "Milano Label Srl",
                "product_name": "Dokuma etiket / woven garment labels",
                "destination_country": "IT",
            },
        ]
        reply = advisor_mediate_reply(
            "dokuma etiket", "Aylık 2.5 milyon metre", matches
        )
        self.assertIn("NordTextil GmbH", reply)
        self.assertIn("Milano Label Srl", reply)
        self.assertIn("Sizin Aylık 2.5 milyon metre kapasitenize", reply)
        self.assertIn("dokuma etiket ürününüze", reply)
        self.assertIn("eşleştiniz", reply)
        self.assertIn("teklif atın", reply)
        self.assertNotIn("İsterseniz", reply)
        self.assertNotIn("?", reply)
        self.assertNotIn("Subject:", reply)
        self.assertIn("0,028", reply)
        self.assertIn("Merter", reply)

    def test_stock_lot_dictates_sell_channels_not_monthly_capacity(self) -> None:
        from agents.session import extract_capacity, extract_stock, looks_like_stock
        from prompts import is_sell_request

        self.assertTrue(looks_like_stock("500.000 adet etiket"))
        self.assertTrue(looks_like_stock("elimde 500 bin adet stok"))
        self.assertEqual(extract_stock("500.000 adet"), "500 bin adet")
        self.assertIsNone(extract_capacity("500.000 adet"))
        self.assertFalse(looks_like_stock("2.5 milyon mt"))
        self.assertTrue(is_sell_request("bu stoğu hangi toptancıya satarım"))

        session = SessionState(session_id="stk1", product="dokuma etiket")
        intent, _ = classify_intent("500.000 adet etiket", session=session)
        self.assertEqual(intent, "intake")
        reply = continue_intake(session, "500.000 adet etiket")
        self.assertEqual(session.stock, "500 bin adet")
        self.assertIsNone(session.capacity)
        self.assertTrue(reply.startswith("Sizin 500 bin adet stokunuza"))
        self.assertIn("iç piyasa", reply.casefold())
        self.assertIn("teklif atın", reply.casefold())
        self.assertIn("0,028", reply)
        self.assertIn("Merter", reply)
        self.assertNotIn("Aylık 500", reply)
        self.assertNotIn("?", reply)
        self.assertNotIn("isterseniz", reply.casefold())

    def test_draft_request_dumps_firm_specific_mail(self) -> None:
        from agents.session import advisor_answer
        from prompts import is_draft_request

        session = SessionState(
            session_id="draft1",
            product="dokuma etiket",
            capacity="Aylık 2.5 milyon metre",
        )
        self.assertTrue(is_draft_request("bana mail taslağı hazırlarsın"))
        intent, _ = classify_intent("bana mail taslağı hazırlarsın", session=session)
        self.assertEqual(intent, "trade_advisor")
        matches = [
            {
                "organization_name": "NordTextil GmbH",
                "product_name": "Woven clothing labels (dokuma etiket)",
                "destination_country": "DE",
            }
        ]
        reply = advisor_answer("bana mail taslağı hazırlarsın", session, matches)
        self.assertIn("NordTextil GmbH", reply)
        self.assertIn("Subject:", reply)
        self.assertIn("Dear Purchasing Team at NordTextil GmbH", reply)
        self.assertIn("monthly 2.5 million meters", reply.casefold())
        self.assertIn("woven clothing labels", reply.casefold())
        self.assertIn("Betreff:", reply)
        self.assertNotIn("Hangisiyle", reply)
        self.assertNotIn("Pazar giriş adımları", reply)
        self.assertNotIn("?", reply)

    def test_woven_label_query_drops_unrelated_cards(self) -> None:
        from agents.retrieve import (
            filter_matches_for_product,
            item_matches_product,
            retrieval_query,
        )

        session = SessionState(session_id="t6", product="dokuma etiket")
        query = retrieval_query(session, "alıcıları bulabilir misin?", role="buyer")
        self.assertIn("dokuma etiket", query.casefold())
        self.assertIn("woven", query.casefold())
        self.assertNotIn("alıcıları", query.casefold())
        olive = {
            "id": "1",
            "product_name": "Extra virgin olive oil",
            "description": "EU retail",
            "hs_code": "1509.10",
            "similarity": 0.71,
        }
        cotton = {
            "id": "2",
            "product_name": "Knitted cotton apparel",
            "description": "Cotton T-shirts",
            "hs_code": "6109.10",
            "similarity": 0.68,
        }
        carpet = {
            "id": "3",
            "product_name": "Hand-knotted wool carpets",
            "hs_code": "5701.10",
            "similarity": 0.66,
        }
        label = {
            "id": "4",
            "product_name": "Woven clothing labels (dokuma etiket)",
            "description": "Apparel brand labels",
            "hs_code": "5807.10",
            "similarity": 0.81,
        }
        self.assertFalse(item_matches_product(olive, "dokuma etiket"))
        self.assertFalse(item_matches_product(cotton, "dokuma etiket"))
        self.assertFalse(item_matches_product(carpet, "dokuma etiket"))
        self.assertTrue(item_matches_product(label, "dokuma etiket"))
        kept = filter_matches_for_product(
            [olive, cotton, carpet, label], "dokuma etiket"
        )
        self.assertEqual([row["product_name"] for row in kept], [label["product_name"]])

    def test_create_session_isolates_intent_and_capacity(self) -> None:
        from agents.session import create_session, hydrate, session_title

        first = create_session()
        first.product = "dokuma etiket"
        first.capacity = "Aylık 2.5 milyon metre"
        first.market = "Almanya"
        first.append("user", "dokuma etiket")
        first.append("assistant", director_options_reply("dokuma etiket", first.capacity))

        second = create_session()
        self.assertNotEqual(first.session_id, second.session_id)
        self.assertIsNone(second.product)
        self.assertIsNone(second.capacity)
        self.assertIsNone(second.market)
        self.assertEqual(second.messages, [])
        self.assertEqual(session_title(second), "Yeni sohbet")
        self.assertEqual(session_title(first), "Dokuma Etiket - Almanya")

        loaded = hydrate(first.session_id, None)
        self.assertEqual(loaded.product, "dokuma etiket")
        self.assertEqual(loaded.capacity, "Aylık 2.5 milyon metre")

        other = hydrate(second.session_id, [])
        self.assertIsNone(other.product)
        self.assertIsNone(other.capacity)
        self.assertEqual(hydrate(first.session_id, None).capacity, "Aylık 2.5 milyon metre")

    def test_delete_session_removes_intent_and_capacity(self) -> None:
        from agents.session import create_session, delete_session, get_session

        session = create_session()
        session.product = "dokuma etiket"
        session.capacity = "Aylık 2.5 milyon metre"
        sid = session.session_id
        self.assertIsNotNone(get_session(sid))
        self.assertTrue(delete_session(sid))
        self.assertIsNone(get_session(sid))
        self.assertFalse(delete_session(sid))
        self.assertFalse(delete_session(""))

    def test_new_product_clears_prior_product_and_volume(self) -> None:
        session = SessionState(session_id="switch1", last_ask="product")
        continue_intake(session, "dokuma etiket")
        continue_intake(session, "500.000 adet")
        self.assertEqual(session.product, "dokuma etiket")
        self.assertEqual(session.stock, "500 bin adet")

        reply = continue_intake(session, "3D oyuncak baskısı")
        self.assertEqual(session.product, "3D oyuncak baskısı")
        self.assertIsNone(session.stock)
        self.assertIsNone(session.capacity)
        folded = reply.casefold()
        self.assertNotIn("dokuma", folded)
        self.assertNotIn("500", reply)
        self.assertNotIn("woven", folded)
        self.assertNotIn("merter", folded)
        self.assertIn("3d", folded)
        self.assertTrue("etsy" in folded or "handmade" in folded or "hediyelik" in folded)
        self.assertIn("spielwarenmesse", folded)
        self.assertIn("tüv", folded)
        self.assertIn("etkinlik", folded)

    def test_hydrate_last_product_wins_and_wipes_volume(self) -> None:
        history = [
            {"role": "user", "content": "dokuma etiket"},
            {"role": "assistant", "content": director_options_reply("dokuma etiket")},
            {"role": "user", "content": "500.000 adet"},
            {
                "role": "assistant",
                "content": director_options_reply(
                    "dokuma etiket", stock="500 bin adet"
                ),
            },
            {"role": "user", "content": "3D oyuncak baskısı"},
        ]
        state = hydrate("switch-hist", history)
        self.assertEqual(state.product, "3D oyuncak baskısı")
        self.assertIsNone(state.stock)
        self.assertIsNone(state.capacity)

    def test_print3d_filter_keeps_channels_drops_mass_and_fairs(self) -> None:
        from agents.retrieve import (
            enrich_matches_for_product,
            item_matches_product,
            retrieval_query,
        )

        session = SessionState(session_id="3d1", product="3D oyuncak baskısı")
        query = retrieval_query(session, "alıcıları bulabilir misin?", role="buyer")
        self.assertIn("etsy", query.casefold())
        self.assertNotIn("apparel", query.casefold())
        self.assertNotIn("alıcıları", query.casefold())

        woven = {
            "id": "1",
            "organization_name": "NordTextil GmbH",
            "product_name": "Woven clothing labels (dokuma etiket)",
            "description": "Apparel brand labels",
            "hs_code": "5807.10",
            "similarity": 0.81,
        }
        mass = {
            "id": "2",
            "organization_name": "Ravensburger",
            "product_name": "Puzzle and toy manufacturing",
            "description": "Mass toy brand",
            "similarity": 0.77,
        }
        fair = {
            "id": "3",
            "organization_name": "Spielwarenmesse eG",
            "product_name": "Nuremberg toy fair",
            "description": "Trade fair",
            "similarity": 0.74,
        }
        cert = {
            "id": "4",
            "organization_name": "TÜV SÜD",
            "product_name": "Toy safety certification",
            "description": "CE notified body",
            "similarity": 0.70,
        }
        etsy = {
            "id": "5",
            "organization_name": "Etsy",
            "product_name": "Personalized 3D printed toys",
            "description": "Handmade PLA PETG miniatures",
            "similarity": 0.88,
        }
        self.assertFalse(item_matches_product(woven, "3D oyuncak baskısı"))
        self.assertFalse(item_matches_product(mass, "3D oyuncak baskısı"))
        self.assertFalse(item_matches_product(fair, "3D oyuncak baskısı"))
        self.assertFalse(item_matches_product(cert, "3D oyuncak baskısı"))
        self.assertTrue(item_matches_product(etsy, "3D oyuncak baskısı"))

        kept, notes = enrich_matches_for_product(
            [woven, mass, fair, cert, etsy], "3D oyuncak baskısı"
        )
        names = [row.get("organization_name") for row in kept]
        self.assertIn("Etsy", names)
        self.assertIn("Amazon Handmade", names)
        self.assertNotIn("Ravensburger", names)
        self.assertNotIn("Spielwarenmesse eG", names)
        self.assertNotIn("TÜV SÜD", names)
        self.assertNotIn("NordTextil GmbH", names)
        joined_notes = " ".join(notes).casefold()
        self.assertIn("spielwarenmesse", joined_notes)
        self.assertIn("tüv", joined_notes)

    def test_print3d_draft_is_one_recipient_with_materials(self) -> None:
        from agents.session import advisor_draft_reply

        matches = [
            {
                "organization_name": "Etsy",
                "product_name": "Personalized 3D printed toys",
                "destination_country": "US",
            },
            {
                "organization_name": "Amazon Handmade",
                "product_name": "Boutique 3D printed products",
                "destination_country": "US",
            },
        ]
        reply = advisor_draft_reply("3D oyuncak baskısı", matches=matches)
        self.assertIn("Dear Etsy,", reply)
        self.assertIn("Dear Amazon Handmade,", reply)
        self.assertNotIn("Dear Etsy, Amazon", reply)
        self.assertNotIn("Dear [", reply)
        self.assertEqual(reply.count("Dear "), 2)
        self.assertIn("PLA", reply)
        self.assertIn("PETG", reply)
        self.assertIn("0.15", reply)
        self.assertIn("5–7", reply)
        self.assertNotIn("Ravensburger", reply)
        self.assertNotIn("same mill", reply.casefold())

    def test_reply_does_not_echo_raw_query(self) -> None:
        from agents.session import (
            advisor_mediate_reply,
            clean_product_name,
            strip_echoed_query,
        )

        raw = "zeytinyağı için alıcıları bulabilir misin?"
        self.assertEqual(clean_product_name(raw), "zeytinyağı")
        reply = advisor_mediate_reply(raw)
        self.assertFalse(reply.casefold().startswith(raw.casefold()))
        self.assertNotIn("alıcıları bulabilir misin", reply.casefold())
        self.assertIn("zeytinyağı", reply.casefold())
        echoed = strip_echoed_query(raw, raw + "\nSizin zeytinyağı ürününüze uygun olarak.")
        self.assertFalse(echoed.casefold().startswith(raw.casefold()))
        self.assertIn("zeytinyağı ürününüze", echoed.casefold())

    def test_toy_event_notes_only_for_print3d_not_olive(self) -> None:
        from agents.retrieve import enrich_matches_for_product, with_domain_notes
        from agents.session import advisor_mediate_reply

        olive = advisor_mediate_reply("zeytinyağı")
        folded = olive.casefold()
        self.assertNotIn("spielwarenmesse", folded)
        self.assertNotIn("tüv", folded)
        self.assertNotIn("etkinlik / sertifika", folded)

        toy = advisor_mediate_reply("3D oyuncak baskısı")
        self.assertIn("spielwarenmesse", toy.casefold())
        self.assertIn("tüv", toy.casefold())

        tuv = {
            "id": "9",
            "organization_name": "TÜV SÜD",
            "product_name": "Certified extra virgin olive oil",
            "description": "CE certification body",
            "hs_code": "1509.10",
            "similarity": 0.8,
        }
        kept, notes = enrich_matches_for_product([tuv], "zeytinyağı")
        names = [row.get("organization_name") for row in kept]
        self.assertNotIn("TÜV SÜD", names)
        attached = with_domain_notes("Zeytinyağı için Almanya notu.", notes, product="zeytinyağı")
        self.assertNotIn("spielwarenmesse", attached.casefold())
        self.assertNotIn("tüv", attached.casefold())
        self.assertNotIn("Etkinlik / Sertifika notu", attached)

    def test_metal_celik_skips_requestion_and_opens_analysis(self) -> None:
        from agents.retrieve import (
            effective_match_threshold,
            enrich_matches_for_product,
            item_matches_product,
            retrieval_query,
        )
        from agents.session import product_kind
        from prompts import is_general_intake

        self.assertFalse(is_small_talk("metal çelik"))
        self.assertFalse(is_general_intake("metal çelik"))
        self.assertEqual(product_kind("metal çelik"), "steel")
        intent, _ = classify_intent("metal çelik")
        self.assertEqual(intent, "intake")

        session = SessionState(session_id="steel1", last_ask="product")
        session.append("assistant", INTAKE_REPLY)
        reply = continue_intake(session, "metal çelik")
        self.assertEqual(session.product, "metal çelik")
        self.assertIsNone(session.last_ask)
        self.assertNotEqual(reply, INTAKE_REPLY)
        self.assertNotIn("hangi ürünü", reply.casefold())
        self.assertNotIn("?", reply)
        self.assertIn("analiz ediyorum", reply.casefold())
        self.assertIn("metal çelik üretimi", reply.casefold())
        self.assertIn("inşaat", reply.casefold())
        self.assertIn("otomotiv", reply.casefold())
        self.assertIn("makine", reply.casefold())
        self.assertIn("tüccar", reply.casefold())

        query = retrieval_query(session, "metal çelik", role="advisor")
        self.assertIn("steel", query.casefold())
        self.assertNotIn("apparel", query.casefold())
        self.assertNotIn("ready-wear", query.casefold())
        self.assertLessEqual(effective_match_threshold("metal çelik", 0.45), 0.28)

        coils = {
            "id": "s1",
            "organization_name": "Black Sea Steel",
            "product_name": "Hot rolled steel coils",
            "description": "Hot-rolled non-alloy steel coils for construction.",
            "hs_code": "7208.10",
            "similarity": 0.31,
        }
        olive = {
            "id": "s2",
            "product_name": "Extra virgin olive oil",
            "hs_code": "1509.10",
            "similarity": 0.71,
        }
        woven = {
            "id": "s3",
            "organization_name": "NordTextil GmbH",
            "product_name": "Woven clothing labels (dokuma etiket)",
            "hs_code": "5807.10",
            "similarity": 0.81,
        }
        gearbox = {
            "id": "s4",
            "product_name": "Automotive transmission parts",
            "description": "Gearbox components for passenger vehicle assembly.",
            "hs_code": "8708.40",
            "similarity": 0.4,
        }
        cnc = {
            "id": "s5",
            "product_name": "CNC machining centers",
            "description": "Vertical CNC machining centers for metalworking SMEs.",
            "hs_code": "8457.10",
            "similarity": 0.4,
        }
        self.assertTrue(item_matches_product(coils, "metal çelik"))
        self.assertFalse(item_matches_product(olive, "metal çelik"))
        self.assertFalse(item_matches_product(woven, "metal çelik"))
        self.assertFalse(item_matches_product(gearbox, "metal çelik"))
        self.assertFalse(item_matches_product(cnc, "metal çelik"))

        cards, notes = enrich_matches_for_product([olive, woven, coils], "metal çelik")
        names = [row["organization_name"] for row in cards]
        self.assertIn("Avrupa inşaat yüklenicileri", names)
        self.assertIn("Otomotiv üreticileri ve tedarikçileri", names)
        self.assertIn("Makine imalatçıları", names)
        self.assertIn("Çelik tüccarları", names)
        self.assertIn("Black Sea Steel", names)
        self.assertNotIn("NordTextil GmbH", names)
        self.assertEqual(notes, [])

    def test_greeting_then_metal_celik_does_not_reask(self) -> None:
        session = SessionState(session_id="steel-greet", last_ask="name")
        intent, _ = classify_intent("metal çelik", session=session)
        self.assertEqual(intent, "intake")
        reply = continue_intake(session, "metal çelik")
        self.assertEqual(session.product, "metal çelik")
        self.assertNotEqual(reply, INTAKE_REPLY)
        self.assertNotIn("hangi ürünü", reply.casefold())
        self.assertIn("analiz ediyorum", reply.casefold())


if __name__ == "__main__":
    unittest.main()
