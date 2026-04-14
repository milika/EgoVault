"""Tests for GmailApiAdapter and gmail_auth helpers.

The Gmail API and OAuth are fully mocked — no network calls, no credentials.
"""
from __future__ import annotations

import base64
import email as _stdlib_email
import email.mime.text
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from egovault.adapters.gmail_api import GmailApiAdapter
from egovault.core.schema import NormalizedRecord
from egovault.core.store import VaultStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store() -> VaultStore:
    s = VaultStore(":memory:")
    s.init_db()
    yield s
    s.close()


@pytest.fixture()
def adapter(store: VaultStore) -> GmailApiAdapter:
    return GmailApiAdapter(store=store)


def _make_raw_message(
    subject: str = "Hello",
    from_: str = "Alice <alice@example.com>",
    to: str = "me@gmail.com",
    date: str = "Mon, 05 Jan 2026 09:00:00 +0000",
    body: str = "Hello from Gmail API",
    thread_id: str = "thread001",
    message_id: str = "<msg001@example.com>",
) -> dict:
    """Return a fake Gmail API 'get' response with a base64url-encoded raw message."""
    msg = _stdlib_email.message.Message()
    msg["From"] = from_
    msg["To"] = to
    msg["Subject"] = subject
    msg["Date"] = date
    msg["Message-ID"] = message_id
    msg.set_payload(body)

    raw_bytes = msg.as_bytes()
    raw_b64 = base64.urlsafe_b64encode(raw_bytes).decode("ascii")
    return {
        "id": message_id.strip("<>"),
        "threadId": thread_id,
        "raw": raw_b64,
    }


def _make_service(messages: list[dict]) -> MagicMock:
    """Return a mock Gmail service that yields *messages* from users.messages.list/get."""
    service = MagicMock()

    # users().messages().list() — return all IDs on the first page, no next page
    list_result = {
        "messages": [{"id": m["id"], "threadId": m["threadId"]} for m in messages]
    }
    service.users().messages().list().execute.return_value = list_result

    # users().messages().get(userId=..., id=id, format="raw") — return matching raw msg
    def _get_execute(userId, id, format):  # noqa: A002
        for m in messages:
            if m["id"] == id:
                return m
        return {}

    service.users().messages().get.side_effect = lambda **kw: MagicMock(
        execute=lambda: _get_execute(**kw)
    )

    return service


# ---------------------------------------------------------------------------
# can_handle() — always False
# ---------------------------------------------------------------------------

class TestCanHandle:
    def test_always_false(self, adapter: GmailApiAdapter, tmp_path: Path) -> None:
        assert adapter.can_handle(tmp_path) is False

    def test_false_for_token_file(self, adapter: GmailApiAdapter, tmp_path: Path) -> None:
        f = tmp_path / "gmail_token.json"
        f.write_text("{}")
        assert adapter.can_handle(f) is False


# ---------------------------------------------------------------------------
# ingest_from_api() with mocked credentials + service
# ---------------------------------------------------------------------------

class TestIngestFromApi:
    def _run(
        self,
        adapter: GmailApiAdapter,
        messages: list[dict],
        query: str = "",
        max_results: int = 500,
    ) -> list[NormalizedRecord]:
        mock_creds = MagicMock()
        mock_service = _make_service(messages)

        with (
            patch("egovault.utils.gmail_auth.load_credentials", return_value=mock_creds),
            patch("egovault.utils.gmail_auth.build_service", return_value=mock_service),
        ):
            return list(
                adapter.ingest_from_api(
                    token_path=Path("fake_token.json"),
                    query=query,
                    max_results=max_results,
                )
            )

    def test_yields_one_record_per_message(self, adapter: GmailApiAdapter) -> None:
        msgs = [
            _make_raw_message("Subject A", message_id="<m1@x.com>", thread_id="t1"),
            _make_raw_message("Subject B", message_id="<m2@x.com>", thread_id="t2"),
        ]
        records = self._run(adapter, msgs)
        assert len(records) == 2

    def test_platform_is_gmail(self, adapter: GmailApiAdapter) -> None:
        records = self._run(adapter, [_make_raw_message()])
        assert records[0].platform == "gmail"

    def test_record_type_is_message(self, adapter: GmailApiAdapter) -> None:
        records = self._run(adapter, [_make_raw_message()])
        assert records[0].record_type == "message"

    def test_sender_id_extracted(self, adapter: GmailApiAdapter) -> None:
        records = self._run(adapter, [_make_raw_message(from_="Bob <bob@test.com>")])
        assert records[0].sender_id == "bob@test.com"

    def test_sender_name_extracted(self, adapter: GmailApiAdapter) -> None:
        records = self._run(adapter, [_make_raw_message(from_="Bob <bob@test.com>")])
        assert records[0].sender_name == "Bob"

    def test_thread_id_from_api_field(self, adapter: GmailApiAdapter) -> None:
        records = self._run(adapter, [_make_raw_message(thread_id="threadXYZ")])
        assert records[0].thread_id == "threadXYZ"

    def test_subject_becomes_thread_name(self, adapter: GmailApiAdapter) -> None:
        records = self._run(adapter, [_make_raw_message(subject="Meeting notes")])
        assert records[0].thread_name == "Meeting notes"

    def test_re_prefix_stripped_from_thread_name(self, adapter: GmailApiAdapter) -> None:
        records = self._run(adapter, [_make_raw_message(subject="Re: Meeting notes")])
        assert records[0].thread_name == "Meeting notes"

    def test_body_is_populated(self, adapter: GmailApiAdapter) -> None:
        records = self._run(adapter, [_make_raw_message(body="Important content here")])
        assert "Important content here" in records[0].body

    def test_timestamp_is_timezone_aware(self, adapter: GmailApiAdapter) -> None:
        records = self._run(adapter, [_make_raw_message()])
        assert records[0].timestamp.tzinfo is not None

    def test_mime_type_is_rfc822(self, adapter: GmailApiAdapter) -> None:
        records = self._run(adapter, [_make_raw_message()])
        assert records[0].mime_type == "message/rfc822"

    def test_raw_contains_api_thread_id(self, adapter: GmailApiAdapter) -> None:
        records = self._run(adapter, [_make_raw_message(thread_id="abc123")])
        assert records[0].raw["api_thread_id"] == "abc123"

    def test_raw_contains_api_message_id(self, adapter: GmailApiAdapter) -> None:
        records = self._run(adapter, [_make_raw_message(message_id="<myid@x.com>")])
        assert records[0].raw["api_message_id"] == "myid@x.com"

    def test_skips_empty_body_messages(self, adapter: GmailApiAdapter) -> None:
        records = self._run(adapter, [_make_raw_message(body="")])
        assert len(records) == 0

    def test_progress_callback_called(self, adapter: GmailApiAdapter) -> None:
        calls: list[int] = []
        mock_creds = MagicMock()
        mock_service = _make_service([
            _make_raw_message("A", message_id="<a@x.com>", thread_id="t1"),
            _make_raw_message("B", message_id="<b@x.com>", thread_id="t2"),
        ])
        with (
            patch("egovault.utils.gmail_auth.load_credentials", return_value=mock_creds),
            patch("egovault.utils.gmail_auth.build_service", return_value=mock_service),
        ):
            list(
                adapter.ingest_from_api(
                    token_path=Path("fake_token.json"),
                    progress_callback=calls.append,
                )
            )
        assert calls == [1, 2]

    def test_raises_if_not_authenticated(self, adapter: GmailApiAdapter) -> None:
        with patch("egovault.utils.gmail_auth.load_credentials", return_value=None):
            with pytest.raises(RuntimeError, match="Not authenticated"):
                list(
                    adapter.ingest_from_api(token_path=Path("no_token.json"))
                )

    def test_default_query_excludes_spam(self, adapter: GmailApiAdapter) -> None:
        """Verify the default query string is passed to the list call."""
        mock_creds = MagicMock()
        mock_service = _make_service([])
        with (
            patch("egovault.utils.gmail_auth.load_credentials", return_value=mock_creds),
            patch("egovault.utils.gmail_auth.build_service", return_value=mock_service),
        ):
            list(adapter.ingest_from_api(token_path=Path("fake_token.json")))

        call_kwargs = mock_service.users().messages().list.call_args.kwargs
        assert "-in:spam" in call_kwargs.get("q", "")
        assert "-in:trash" in call_kwargs.get("q", "")

    def test_custom_query_forwarded(self, adapter: GmailApiAdapter) -> None:
        mock_creds = MagicMock()
        mock_service = _make_service([])
        with (
            patch("egovault.utils.gmail_auth.load_credentials", return_value=mock_creds),
            patch("egovault.utils.gmail_auth.build_service", return_value=mock_service),
        ):
            list(
                adapter.ingest_from_api(
                    token_path=Path("fake_token.json"),
                    query="in:inbox is:important",
                )
            )

        call_kwargs = mock_service.users().messages().list.call_args.kwargs
        assert call_kwargs["q"] == "in:inbox is:important"


# ---------------------------------------------------------------------------
# API failures are handled gracefully (not raised, just skipped)
# ---------------------------------------------------------------------------

class TestApiFailureHandling:
    def test_failed_get_skips_message(self, adapter: GmailApiAdapter) -> None:
        mock_creds = MagicMock()
        mock_service = MagicMock()

        # list() returns one message ID
        mock_service.users().messages().list().execute.return_value = {
            "messages": [{"id": "bad_id", "threadId": "t1"}]
        }
        # get() raises an exception
        mock_service.users().messages().get.side_effect = Exception("API error")

        with (
            patch("egovault.utils.gmail_auth.load_credentials", return_value=mock_creds),
            patch("egovault.utils.gmail_auth.build_service", return_value=mock_service),
        ):
            records = list(adapter.ingest_from_api(token_path=Path("fake_token.json")))

        assert records == []


# ---------------------------------------------------------------------------
# Deduplication via VaultStore
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_reimport_produces_no_new_records(
        self, adapter: GmailApiAdapter, store: VaultStore
    ) -> None:
        messages = [_make_raw_message("Hello", message_id="<dup@x.com>", thread_id="t1")]
        mock_creds = MagicMock()
        mock_service = _make_service(messages)

        with (
            patch("egovault.utils.gmail_auth.load_credentials", return_value=mock_creds),
            patch("egovault.utils.gmail_auth.build_service", return_value=mock_service),
        ):
            first = list(adapter.ingest_from_api(token_path=Path("fake_token.json")))
            for rec in first:
                store.upsert_record(rec)

            second = list(adapter.ingest_from_api(token_path=Path("fake_token.json")))
            new_inserts = sum(1 for rec in second if store.upsert_record(rec))

        assert new_inserts == 0


# ---------------------------------------------------------------------------
# gmail_auth — _require_deps auto-installs and raises on pip failure
# ---------------------------------------------------------------------------

class TestGmailAuthRequireDeps:
    def test_auto_install_attempted_when_deps_missing(self) -> None:
        """When deps are missing, _require_deps() calls pip before raising."""
        from unittest.mock import MagicMock, patch

        # Simulate packages being absent by making _find_missing() return them.
        fake_proc = MagicMock()
        fake_proc.returncode = 1  # simulate pip failure
        fake_proc.stderr = "error: could not find packages"

        with (
            patch("egovault.utils.gmail_auth._find_missing", return_value=["google-auth"]),
            patch("egovault.utils.gmail_auth.subprocess.run", return_value=fake_proc),
        ):
            from egovault.utils import gmail_auth
            with pytest.raises(ImportError, match="pip install egovault"):
                gmail_auth._require_deps()

    def test_no_install_when_deps_present(self) -> None:
        """When all deps are installed, _require_deps() is a no-op."""
        from unittest.mock import patch

        with (
            patch("egovault.utils.gmail_auth._find_missing", return_value=[]),
            patch("egovault.utils.gmail_auth.subprocess.run") as mock_run,
        ):
            from egovault.utils import gmail_auth
            gmail_auth._require_deps()  # should not raise
            mock_run.assert_not_called()

    def test_get_token_path(self, tmp_path: Path) -> None:
        from egovault.utils.gmail_auth import TOKEN_FILENAME, get_token_path
        assert get_token_path(tmp_path) == tmp_path / TOKEN_FILENAME

    def test_load_credentials_returns_none_when_no_file(self, tmp_path: Path) -> None:
        from egovault.utils.gmail_auth import load_credentials
        try:
            result = load_credentials(tmp_path / "missing_token.json")
            assert result is None
        except ImportError:
            pytest.skip("google-auth-oauthlib not installed")


class TestRunOauthFlowEmbedded:
    """run_oauth_flow_embedded() uses from_client_config instead of a JSON file."""

    def test_calls_from_client_config_not_from_file(self, tmp_path: Path) -> None:
        mock_creds = MagicMock()
        mock_creds.to_json.return_value = '{"token": "test"}'

        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = mock_creds

        mock_flow_cls = MagicMock()
        mock_flow_cls.from_client_config.return_value = mock_flow

        with (
            patch("egovault.utils.gmail_auth._require_deps"),
            patch("google_auth_oauthlib.flow.InstalledAppFlow", mock_flow_cls),
        ):
            from egovault.utils import gmail_auth
            gmail_auth.run_oauth_flow_embedded(
                "my-client-id.apps.googleusercontent.com",
                "my-secret",
                tmp_path / "token.json",
            )
            mock_flow_cls.from_client_config.assert_called_once()
            call_args = mock_flow_cls.from_client_config.call_args
            config = call_args[0][0]
            assert config["installed"]["client_id"] == "my-client-id.apps.googleusercontent.com"
            assert config["installed"]["client_secret"] == "my-secret"
            assert "auth_uri" in config["installed"]
            assert "token_uri" in config["installed"]

    def test_token_file_written(self, tmp_path: Path) -> None:
        mock_creds = MagicMock()
        mock_creds.to_json.return_value = '{"token": "abc"}'
        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = mock_creds
        mock_flow_cls = MagicMock()
        mock_flow_cls.from_client_config.return_value = mock_flow

        with (
            patch("egovault.utils.gmail_auth._require_deps"),
            patch("google_auth_oauthlib.flow.InstalledAppFlow", mock_flow_cls),
        ):
            from egovault.utils import gmail_auth
            token_path = tmp_path / "token.json"
            gmail_auth.run_oauth_flow_embedded("cid", "csecret", token_path)
            assert token_path.exists()
            assert token_path.read_text() == '{"token": "abc"}'

    def test_run_local_server_called_with_port_zero(self, tmp_path: Path) -> None:
        mock_creds = MagicMock()
        mock_creds.to_json.return_value = "{}"
        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = mock_creds
        mock_flow_cls = MagicMock()
        mock_flow_cls.from_client_config.return_value = mock_flow

        with (
            patch("egovault.utils.gmail_auth._require_deps"),
            patch("google_auth_oauthlib.flow.InstalledAppFlow", mock_flow_cls),
        ):
            from egovault.utils import gmail_auth
            gmail_auth.run_oauth_flow_embedded("cid", "cs", tmp_path / "t.json")
            mock_flow.run_local_server.assert_called_once_with(port=0, open_browser=True)


class TestGmailOAuthSettingsConfig:
    """GmailOAuthSettings is read from toml [gmail] section and env vars."""

    def test_default_values_are_empty(self) -> None:
        from egovault.config import GmailOAuthSettings
        cfg = GmailOAuthSettings()
        assert cfg.client_id == ""
        assert cfg.client_secret == ""

    def test_load_from_toml(self, tmp_path: Path) -> None:
        import tomllib  # noqa: F401 — just confirm it exists; file is written as text
        from egovault.config import load_settings
        cfg_file = tmp_path / "egovault.toml"
        cfg_file.write_text(
            '[gmail]\nclient_id = "test-id"\nclient_secret = "test-secret"\n',
            encoding="utf-8",
        )
        settings = load_settings(cfg_file)
        assert settings.gmail.client_id == "test-id"
        assert settings.gmail.client_secret == "test-secret"

    def test_env_vars_override_toml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from egovault.config import load_settings
        cfg_file = tmp_path / "egovault.toml"
        cfg_file.write_text(
            '[gmail]\nclient_id = "from-toml"\nclient_secret = "from-toml-secret"\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("EGOVAULT_GMAIL_CLIENT_ID", "from-env")
        monkeypatch.setenv("EGOVAULT_GMAIL_CLIENT_SECRET", "from-env-secret")
        settings = load_settings(cfg_file)
        assert settings.gmail.client_id == "from-env"
        assert settings.gmail.client_secret == "from-env-secret"

    def test_empty_when_neither_toml_nor_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from egovault.config import load_settings
        monkeypatch.delenv("EGOVAULT_GMAIL_CLIENT_ID", raising=False)
        monkeypatch.delenv("EGOVAULT_GMAIL_CLIENT_SECRET", raising=False)
        cfg_file = tmp_path / "empty.toml"
        cfg_file.write_text("", encoding="utf-8")
        settings = load_settings(cfg_file)
        assert settings.gmail.client_id == ""
        assert settings.gmail.client_secret == ""

