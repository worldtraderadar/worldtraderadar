"""Phase 2 — auth/me + speak Authorization expectations."""

from __future__ import annotations

import os
import unittest
from uuid import uuid4

os.environ.setdefault("WTR_JWT_SECRET", "wtr-test-jwt-secret-key-32b!!xx")
os.environ.setdefault("SUPABASE_JWT_SECRET", os.environ["WTR_JWT_SECRET"])
os.environ.setdefault("SUPABASE_JWT_ISSUER", "http://localhost/auth/v1")
os.environ.setdefault("SUPABASE_JWT_AUD", "authenticated")
os.environ.setdefault("NEXT_PUBLIC_SUPABASE_URL", "http://localhost")

from fastapi.testclient import TestClient

from auth import mint_test_token, reset_membership_memory, seed_membership
import main as main_mod


def _auth(user_id: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {mint_test_token(user_id)}"}


class Phase2AuthMeTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_membership_memory()
        self.user = str(uuid4())
        self.account = str(uuid4())
        seed_membership(
            user_id=self.user,
            account_id=self.account,
            account_slug="tenant-phase2",
            role="owner",
            plan_id="free",
        )
        self.client = TestClient(main_mod.app)

    def tearDown(self) -> None:
        reset_membership_memory()

    def test_auth_me_requires_bearer(self) -> None:
        r = self.client.get("/auth/me")
        self.assertEqual(r.status_code, 401)

    def test_auth_me_returns_trusted_account(self) -> None:
        r = self.client.get("/auth/me", headers=_auth(self.user))
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["user_id"], self.user)
        self.assertEqual(body["account_id"], self.account)
        self.assertEqual(body["account_slug"], "tenant-phase2")
        self.assertEqual(body["plan_id"], "free")
        self.assertEqual(body["role"], "owner")

    def test_auth_me_ignores_slug_header(self) -> None:
        r = self.client.get(
            "/auth/me",
            headers={
                **_auth(self.user),
                "X-Account-Slug": "other-tenant",
            },
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["account_slug"], "tenant-phase2")

    def test_speak_requires_auth(self) -> None:
        r = self.client.post("/speak", json={"text": "Merhaba", "lang": "tr"})
        self.assertEqual(r.status_code, 401)


if __name__ == "__main__":
    unittest.main()
