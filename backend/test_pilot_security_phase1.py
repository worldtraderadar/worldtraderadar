"""Pilot Security Phase 1 — negative authorization tests."""

from __future__ import annotations

import os
import unittest
from uuid import uuid4

os.environ.setdefault("WTR_JWT_SECRET", "wtr-test-jwt-secret-key-32b!!")
os.environ.setdefault("SUPABASE_JWT_SECRET", os.environ["WTR_JWT_SECRET"])
os.environ.setdefault("SUPABASE_JWT_ISSUER", "http://localhost/auth/v1")
os.environ.setdefault("SUPABASE_JWT_AUD", "authenticated")
os.environ.setdefault("NEXT_PUBLIC_SUPABASE_URL", "http://localhost")

from fastapi.testclient import TestClient

from agents.session import create_session, list_sessions
from auth import (
    mint_test_token,
    reset_membership_memory,
    seed_membership,
    verify_access_token,
)
from auth.jwt_verify import AuthError
from auth.policy import policy_for_plan
import main as main_mod


def _auth_header(user_id: str) -> dict[str, str]:
    token = mint_test_token(user_id)
    return {"Authorization": f"Bearer {token}"}


class JwtVerificationTests(unittest.TestCase):
    def test_valid_token_yields_sub(self) -> None:
        uid = str(uuid4())
        token = mint_test_token(uid)
        claims = verify_access_token(token)
        self.assertEqual(claims["sub"], uid)

    def test_expired_token_rejected(self) -> None:
        uid = str(uuid4())
        token = mint_test_token(uid, ttl_seconds=-10)
        with self.assertRaises(AuthError):
            verify_access_token(token)

    def test_wrong_audience_rejected(self) -> None:
        uid = str(uuid4())
        token = mint_test_token(uid, audience="wrong-aud")
        with self.assertRaises(AuthError):
            verify_access_token(token)

    def test_plan_policy_foundation(self) -> None:
        free = policy_for_plan("free")
        pro = policy_for_plan("pro")
        self.assertFalse(free.can_view_company_contacts)
        self.assertTrue(pro.can_view_company_contacts)
        self.assertEqual(free.max_images, 0)
        self.assertFalse(free.can_upload_images)


class PilotSecurityPhase1Tests(unittest.TestCase):
    def setUp(self) -> None:
        reset_membership_memory()
        self.user_a = str(uuid4())
        self.user_b = str(uuid4())
        self.acc_a = str(uuid4())
        self.acc_b = str(uuid4())
        seed_membership(
            user_id=self.user_a,
            account_id=self.acc_a,
            account_slug="tenant-a",
            role="owner",
            plan_id="free",
        )
        seed_membership(
            user_id=self.user_b,
            account_id=self.acc_b,
            account_slug="tenant-b",
            role="owner",
            plan_id="pro",
        )
        # Clear session store between tests via creating isolated IDs only.
        self.client = TestClient(main_mod.app)
        # Avoid real Supabase during ownership-only HTTP tests.
        main_mod.app.state.supabase = None
        main_mod.app.state.supabase_error = "test"
        main_mod.app.state.supabase_lock = __import__("asyncio").Lock()
        if not hasattr(main_mod.app.state, "http") or main_mod.app.state.http is None:
            import httpx

            main_mod.app.state.http = httpx.AsyncClient()

    def tearDown(self) -> None:
        reset_membership_memory()

    def test_01_unauthenticated_consult_401(self) -> None:
        r = self.client.post("/consult", json={"question": "Merhaba"})
        self.assertEqual(r.status_code, 401, r.text)

    def test_02_unauthenticated_consult_stream_401(self) -> None:
        r = self.client.post("/consult/stream", json={"question": "Merhaba"})
        self.assertEqual(r.status_code, 401, r.text)

    def test_03_cross_account_session_get_404(self) -> None:
        b_session = create_session(
            account_id=self.acc_b, created_by_user_id=self.user_b
        )
        b_session.product = "etiket"
        r = self.client.get(
            f"/consult/sessions/{b_session.session_id}",
            headers=_auth_header(self.user_a),
        )
        self.assertEqual(r.status_code, 404, r.text)

    def test_04_cross_account_session_delete_404(self) -> None:
        b_session = create_session(
            account_id=self.acc_b, created_by_user_id=self.user_b
        )
        b_session.product = "etiket"
        r = self.client.delete(
            f"/consult/sessions/{b_session.session_id}",
            headers=_auth_header(self.user_a),
        )
        self.assertEqual(r.status_code, 404, r.text)
        # Still present for owner
        r2 = self.client.get(
            f"/consult/sessions/{b_session.session_id}",
            headers=_auth_header(self.user_b),
        )
        self.assertEqual(r2.status_code, 200, r2.text)

    def test_05_cross_account_consult_404(self) -> None:
        b_session = create_session(
            account_id=self.acc_b, created_by_user_id=self.user_b
        )
        b_session.product = "etiket"
        r = self.client.post(
            "/consult",
            headers=_auth_header(self.user_a),
            json={
                "question": "Merhaba",
                "session_id": b_session.session_id,
            },
        )
        self.assertEqual(r.status_code, 404, r.text)

    def test_06_session_list_isolated(self) -> None:
        a_session = create_session(
            account_id=self.acc_a, created_by_user_id=self.user_a
        )
        a_session.product = "pamuk"
        b_session = create_session(
            account_id=self.acc_b, created_by_user_id=self.user_b
        )
        b_session.product = "zeytin"
        r = self.client.get("/consult/sessions", headers=_auth_header(self.user_a))
        self.assertEqual(r.status_code, 200, r.text)
        ids = {row["session_id"] for row in r.json()}
        self.assertIn(a_session.session_id, ids)
        self.assertNotIn(b_session.session_id, ids)

    def test_07_x_account_slug_spoof_ignored(self) -> None:
        r = self.client.get(
            "/billing/me",
            headers={
                **_auth_header(self.user_a),
                "X-Account-Slug": "tenant-b",
            },
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["slug"], "tenant-a")
        self.assertNotEqual(body["slug"], "tenant-b")

    def test_08_free_user_cannot_gain_pro_via_slug(self) -> None:
        from billing.redact import redact_record

        # Simulate request.state.auth free plan even if header says tenant-b/pro.
        class _Req:
            state = type(
                "S",
                (),
                {
                    "auth": type(
                        "P",
                        (),
                        {
                            "plan_id": "free",
                            "account_slug": "tenant-a",
                            "account_id": self.acc_a,
                        },
                    )(),
                    "account_slug": "tenant-a",
                },
            )()

        # Direct redact uses plan_id_for_request async — unit-check policy + redact.
        row = {
            "contact_email": "secret@b.example",
            "website": "https://b.example",
            "tax_id": "TR1",
            "quantity": 99,
            "value_usd": 1000,
        }
        redacted = redact_record(row, "free")
        self.assertIsNone(redacted["contact_email"])
        self.assertTrue(redacted["locked"]["contact"])

        r = self.client.get(
            "/billing/me",
            headers={
                **_auth_header(self.user_a),
                "X-Account-Slug": "tenant-b",
            },
        )
        self.assertEqual(r.json()["plan_id"], "free")
        self.assertEqual(r.json()["slug"], "tenant-a")

    def test_09_history_scoped_empty_without_foreign(self) -> None:
        # Without Supabase, history returns []; must not 500 and not leak.
        r = self.client.get(
            "/consult/history", headers=_auth_header(self.user_a)
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIsInstance(r.json(), list)

    def test_10_agent_logs_scoped(self) -> None:
        r = self.client.get("/agent-logs", headers=_auth_header(self.user_a))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIsInstance(r.json(), list)

    def test_11_agent_runs_scoped(self) -> None:
        r = self.client.get("/agent-runs", headers=_auth_header(self.user_a))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIsInstance(r.json(), list)

    def test_12_billing_slug_spoof_no_cross_account(self) -> None:
        r = self.client.get(
            "/billing/me",
            headers={
                **_auth_header(self.user_a),
                "X-Account-Slug": "tenant-b",
            },
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["slug"], "tenant-a")
        # Plan change applies to trusted account only (A), not B.
        r2 = self.client.post(
            "/billing/plan",
            headers={
                **_auth_header(self.user_a),
                "X-Account-Slug": "tenant-b",
            },
            json={"plan_id": "pro"},
        )
        self.assertEqual(r2.status_code, 200, r2.text)
        self.assertEqual(r2.json()["slug"], "tenant-a")
        # B still its own billing identity when authenticated as B
        r3 = self.client.get("/billing/me", headers=_auth_header(self.user_b))
        self.assertEqual(r3.json()["slug"], "tenant-b")

    def test_search_requires_auth(self) -> None:
        r = self.client.post("/search", json={"query": "etiket"})
        self.assertEqual(r.status_code, 401)

    def test_owner_can_read_own_session(self) -> None:
        s = create_session(account_id=self.acc_a, created_by_user_id=self.user_a)
        s.product = "dokuma"
        r = self.client.get(
            f"/consult/sessions/{s.session_id}",
            headers=_auth_header(self.user_a),
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["session_id"], s.session_id)

    def test_list_sessions_helper_filters(self) -> None:
        create_session(account_id=self.acc_a, created_by_user_id=self.user_a).product = "a"
        create_session(account_id=self.acc_b, created_by_user_id=self.user_b).product = "b"
        only_a = list_sessions(account_id=self.acc_a)
        self.assertTrue(all(row.get("account_id") == self.acc_a for row in only_a))


if __name__ == "__main__":
    unittest.main()
