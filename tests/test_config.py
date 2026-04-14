"""Unit tests for config module — load_settings, reset_settings, load_agent_prompts."""
from __future__ import annotations

from pathlib import Path


from egovault.config import get_settings, load_agent_prompts, load_settings, reset_settings


class TestResetSettings:
    def setup_method(self) -> None:
        reset_settings()

    def teardown_method(self) -> None:
        reset_settings()

    def test_reset_allows_reload(self, tmp_path: Path) -> None:
        toml = tmp_path / "egovault.toml"
        toml.write_text('[general]\nvault_db = "first.db"\n', encoding="utf-8")
        s1 = load_settings(toml)
        reset_settings()
        toml.write_text('[general]\nvault_db = "second.db"\n', encoding="utf-8")
        s2 = load_settings(toml)
        assert str(s1.vault_db) != str(s2.vault_db)

    def test_default_settings_returned_without_config(self) -> None:
        settings = load_settings(Path("/nonexistent/path.toml"))
        assert settings is not None
        assert settings.vault_db is not None


class TestLoadSettings:
    def setup_method(self) -> None:
        reset_settings()

    def teardown_method(self) -> None:
        reset_settings()

    def test_loads_llm_model(self, tmp_path: Path) -> None:
        toml = tmp_path / "egovault.toml"
        toml.write_text('[llm]\nmodel = "llama3"\n', encoding="utf-8")
        settings = load_settings(toml)
        assert settings.llm.model == "llama3"

    def test_loads_vault_db_path(self, tmp_path: Path) -> None:
        toml = tmp_path / "egovault.toml"
        # Use forward-slash path so TOML parses without escape issues
        toml.write_text('[general]\nvault_db = "custom.db"\n', encoding="utf-8")
        settings = load_settings(toml)
        assert settings.vault_db == "custom.db"

    def test_get_settings_cached_on_second_call(self) -> None:
        # get_settings() (not load_settings) provides the singleton cache
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_missing_file_returns_defaults(self) -> None:
        settings = load_settings(Path("/no/such/file.toml"))
        assert settings.llm is not None

    def test_missing_file_defaults_llama_cpp_provider(self) -> None:
        settings = load_settings(Path("/no/such/file.toml"))
        assert settings.llm.provider == "llama_cpp"
        assert settings.llm.base_url == "http://127.0.0.1:8080"
        assert settings.llama_cpp.n_gpu_layers == 99


class TestLoadAgentPrompts:
    def setup_method(self) -> None:
        reset_settings()  # also clears lru_cache on load_agent_prompts

    def test_returns_enrichment_and_chat_keys(self, tmp_path: Path) -> None:
        prompts = load_agent_prompts(tmp_path / "agent.md")
        assert "enrichment" in prompts
        assert "chat" in prompts

    def test_custom_section_parsed(self, tmp_path: Path) -> None:
        agent_md = tmp_path / "agent.md"
        agent_md.write_text(
            "## enrichment\nMy enrichment prompt.\n\n## chat\nMy chat prompt.\n",
            encoding="utf-8",
        )
        prompts = load_agent_prompts(agent_md)
        assert prompts["enrichment"] == "My enrichment prompt."
        assert prompts["chat"] == "My chat prompt."

    def test_missing_section_falls_back_to_default(self, tmp_path: Path) -> None:
        agent_md = tmp_path / "agent.md"
        agent_md.write_text("## enrichment\nCustom enrichment only.\n", encoding="utf-8")
        prompts = load_agent_prompts(agent_md)
        assert prompts["enrichment"] == "Custom enrichment only."
        assert prompts["chat"]  # should be non-empty default
