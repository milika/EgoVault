"""Configuration loader and settings dataclass for EgoVault."""
from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_DEFAULT_CONFIG_PATHS = [
    Path("data/egovault.toml"),          # personal config — gitignored
    Path.home() / ".config" / "egovault" / "egovault.toml",
]

logger = logging.getLogger(__name__)


@dataclass
class LLMSettings:
    # Provider: "llama_cpp" (llama-server, default) or "openai" / any OpenAI-compatible.
    provider: str = "llama_cpp"
    model: str = "gemma-4-e4b-it"
    base_url: str = "http://127.0.0.1:8080"
    api_key: str = ""                # required for cloud providers
    timeout_seconds: int = 300
    chunk_target_tokens: int = 2000
    # Maximum estimated context tokens before old tool results are summarised.
    # 0 = disabled.  Recommended: 24000 for 32k-ctx models, 8000 for 8k-ctx models.
    max_ctx_tokens: int = 0



@dataclass
class RerankerSettings:
    enabled: bool = True
    # "bm25"           — pure-Python BM25 scoring; no extra dependencies.
    # "cross-encoder"  — sentence-transformers CrossEncoder; much more accurate but
    #                    included in `pip install egovault` (~1-2 GB with torch).
    # "auto"           — try cross-encoder first, silently fall back to bm25.
    backend: str = "auto"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # used by cross-encoder / auto


@dataclass
class LlamaCppSettings:
    """Settings for the llama.cpp llama-server backend.

    Used when ``[llm] provider = "llama_cpp"``.

    When ``manage = false`` (default) llama-server must already be running.
    When ``manage = true`` EgoVault starts the server automatically, computing
    a ``ctx-size`` that keeps total GPU VRAM use at ``vram_budget_pct``
    (default 80 %).

    Typical launch command (manual mode)::

        llama-server -m path/to/model.gguf \\
            --n-gpu-layers 99 --flash-attn --ctx-size 16384 \\
            --host 127.0.0.1 --port 8080 --embedding --pooling mean
    """
    # HTTP base URL of the llama-server instance.
    base_url: str = "http://127.0.0.1:8080"
    # Number of model layers to offload to GPU; 99 = all layers (full GPU).
    n_gpu_layers: int = 99
    # Context size in tokens.  0 = auto-compute from VRAM budget when manage=true,
    # or use the server's own default when manage=false.
    ctx_size: int = 0
    # llama-server also acts as the embedding backend when this is true.
    # It serves POST /v1/embeddings (OpenAI-compatible format).
    embed: bool = True
    # ── Managed-startup settings ──────────────────────────────────────────────
    # When true, EgoVault starts llama-server automatically on each command that
    # needs the LLM.  Requires model_path to be set.
    manage: bool = True
    # Absolute (or ~- or ./-relative) path to the GGUF model file.
    # Required when manage = true.  Relative paths are resolved from the
    # working directory (project root when launched via `ego`).
    model_path: str = "./models/gemma-4-E2B-it-UD-Q4_K_XL.gguf"
    # Enable flash attention (--flash-attn).  Halves KV-cache VRAM, effectively
    # doubling available context for the same memory budget.  Recommended.
    flash_attn: bool = True
    # Target fraction of currently-free GPU VRAM to give to the KV cache.
    # ctx-size is auto-computed as: free_vram × vram_budget_pct / kv_mb_per_token.
    # The model loads unrestricted; only context size is constrained.
    # Default: 0.80 (80 % of free VRAM goes to context).
    vram_budget_pct: float = 0.80
    # HuggingFace repo to auto-download the model from when model_path is missing.
    # Set to "" to disable auto-download.
    # The file downloaded is Path(model_path).name from the repo's main branch.
    model_hf_repo: str = "unsloth/gemma-4-E2B-it-GGUF"
    # Path to the multimodal projector GGUF (--mmproj).  Required for vision /
    # image-understanding features (e.g. image ingestion via `egovault scan`).
    # Set to "" if your model does not support vision.
    # Example: "./models/gemma-4-E2B-it-UD-mmproj-f16.gguf"
    mmproj_path: str = ""
    # HuggingFace filename to auto-download for the mmproj when mmproj_path is
    # missing.  Downloaded from the same model_hf_repo.  Set to "" to disable.
    mmproj_hf_file: str = ""


@dataclass
class EmbeddingSettings:
    # Set enabled = true after running: egovault embed
    # This unlocks semantic (vector) search alongside FTS5 keyword search.
    enabled: bool = False
    # Embedding backend: "llama_cpp" (default, inherits from [llm]) or "openai".
    # Uses POST /v1/embeddings (OpenAI-compatible format) for all backends.
    # When empty, inherits from [llm] provider.
    provider: str = ""
    # Embedding model served by llama-server (or any OpenAI-compatible endpoint).
    # nomic-embed-text (~274 MB) produces 768-dim vectors.
    model: str = "nomic-embed-text"
    # Empty = inherit the [llm] base_url.  Set explicitly to use a different endpoint.
    base_url: str = ""
    # HyDE (Hypothetical Document Embeddings): before embedding a query, ask the
    # LLM to generate a short hypothetical vault passage that would answer it.
    # The passage is embedded instead of the raw question, matching
    # document-to-document rather than question-to-document.
    # Adds ~1 extra LLM call per query.  Requires embeddings.enabled = true.
    hyde_enabled: bool = True
    # Contextual Retrieval: at index time, prepend a short LLM-generated context
    # blurb to each record body before embedding and FTS5 indexing.  This gives
    # the embedding model and BM25 document-level context they would otherwise
    # lack.  Run `egovault context` after enabling to build the prefixes.
    # See docs/rag-improvements.md §5 for details.
    contextual_enabled: bool = True
    # HyPE (Hypothetical Prompt Embeddings): at index time, ask the LLM to
    # generate 3-5 questions a user might ask about each record.  Those
    # questions are embedded and stored in record_question_embeddings.  At
    # retrieval time the user's query is matched against question embeddings
    # (question-to-question similarity is much tighter than
    # question-to-document).  Zero query-time LLM overhead — all cost is at
    # index time.  Run `egovault embed` after enabling to build question
    # embeddings.  See docs/rag-improvements.md §6 for details.
    hype_enabled: bool = True


@dataclass
class CRAGSettings:
    """Corrective Retrieval-Augmented Generation (CRAG-lite) settings.

    When enabled, if all retrieved chunks score below *threshold* after
    Stage-2 reranking, a corrective re-retrieval is triggered using *strategy*.
    See docs/rag-improvements.md §7 for details.
    """
    enabled: bool = True
    # Score below which re-retrieval is triggered.
    # For BM25 backend use ~0.25; for cross-encoder use ~0.10.
    threshold: float = 0.10
    # "hyde"    — re-retrieve with HyDE query augmentation (default)
    # "broaden" — widen the FTS5 query and re-retrieve
    # "empty"   — return nothing and tell the LLM nothing was found
    strategy: str = "hyde"


@dataclass
class SentenceWindowSettings:
    """Sentence Window Retrieval settings (P4).

    When enabled, each vault record is split into overlapping sentence windows
    at index time and each window is embedded separately.  At retrieval time
    the user's query is matched against window embeddings; the winning window
    is expanded by *window_size* sentences in each direction before being
    passed to the LLM as context.  This gives sub-record semantic precision.
    See docs/rag-improvements.md §8 for details.
    """
    enabled: bool = True
    # Number of sentences per window (embedding granularity).
    window_size: int = 3
    # How many chunks to overlap between consecutive windows.
    overlap: int = 1


@dataclass
class WebSearchSettings:
    """Optional web search tool accessible from the EgoVault chat.

    Two backends are supported:

    * ``"duckduckgo"`` (default) — uses the ``duckduckgo-search`` library.
      No API key, no server, no Docker.  Just ``pip install duckduckgo-search``.
      Set ``provider = "duckduckgo"`` and leave ``searxng_url`` empty.

    * ``"searxng"`` — calls any SearXNG instance's JSON API.
      Best with a self-hosted instance (public ones rate-limit bots aggressively).
      Set ``provider = "searxng"`` and ``searxng_url = "http://localhost:8888"``.

    Set ``provider = ""`` (or omit the section) to disable the tool entirely.
    """
    # "duckduckgo" | "searxng" | "" (disabled)
    provider: str = "duckduckgo"
    # SearXNG base URL — only used when provider = "searxng".
    searxng_url: str = ""
    # Fallback SearXNG instances (tried in order on 429/403/timeout).
    fallback_urls: list = field(default_factory=list)
    # Maximum results to return per search (1–20).
    max_results: int = 5
    # Comma-separated SearXNG categories, e.g. "general", "news,general".
    categories: str = "general"


@dataclass
class GmailOAuthSettings:
    """Optional embedded OAuth2 credentials for the Gmail API.

    Set these via ``egovault.toml`` ``[gmail]`` section or the env vars
    ``EGOVAULT_GMAIL_CLIENT_ID`` / ``EGOVAULT_GMAIL_CLIENT_SECRET``.
    When present, users can run ``/gmail-auth`` without supplying a
    ``client_secret_*.json`` file.
    """
    client_id: str = ""
    client_secret: str = ""


@dataclass
class SchedulerSettings:
    """Auto-refresh schedules for active adapters.

    Set ``auto_refresh_inbox_minutes`` and/or ``auto_refresh_gmail_minutes``
    to a positive integer to automatically scan the inbox or sync Gmail at
    that interval when the chat session is running.  Set to 0 to disable.

    These become the *default* schedules registered on first startup.  Users
    can also add/remove schedules interactively with the [bold]/schedule[/bold]
    command in the chat REPL.
    """
    auto_refresh_inbox_minutes: int = 30      # 0 = disabled
    auto_refresh_gmail_minutes: int = 30      # 0 = disabled
    auto_refresh_telegram_minutes: int = 30   # 0 = disabled


@dataclass
class TelegramSettings:
    """Settings for the optional Telegram bot interface.

    Get a bot token from @BotFather on Telegram (free)::

        /newbot  →  copy the token  →  set below

    Find your chat ID by messaging @userinfobot on Telegram.
    Add it to *allowed_chat_ids* so only you can query the vault.
    """
    # Bot token from @BotFather.  Empty = Telegram bot disabled.
    token: str = ""
    # Telegram chat IDs allowed to use the bot (whitelist).
    # Get your ID from @userinfobot.  Leave empty to deny all access.
    allowed_chat_ids: list = field(default_factory=list)
    # Maximum vault records to retrieve per query.
    top_n: int = 10


@dataclass
class Settings:
    vault_db: str = "./data/vault.db"
    output_dir: str = "./output"
    inbox_dir: str = "./inbox"
    log_level: str = "INFO"
    wan_password_hash: str = ""  # sha256:<hex> — required to use --wan
    llm: LLMSettings = field(default_factory=LLMSettings)
    llama_cpp: LlamaCppSettings = field(default_factory=LlamaCppSettings)
    reranker: RerankerSettings = field(default_factory=RerankerSettings)
    embeddings: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    crag: CRAGSettings = field(default_factory=CRAGSettings)
    sentence_window: SentenceWindowSettings = field(default_factory=SentenceWindowSettings)
    gmail: GmailOAuthSettings = field(default_factory=GmailOAuthSettings)
    scheduler: SchedulerSettings = field(default_factory=SchedulerSettings)
    web_search: WebSearchSettings = field(default_factory=WebSearchSettings)
    telegram: TelegramSettings = field(default_factory=TelegramSettings)


_settings: Settings | None = None

_PERSONAL_CONFIG = Path("data/egovault.toml")

_DEFAULT_CONFIG_CONTENT = """\
# EgoVault personal configuration
# Generated on first run — this file is gitignored; edit freely.

[general]
vault_db  = "./data/vault.db"
output_dir = "./output"
inbox_dir  = "./inbox"
log_level  = "INFO"               # DEBUG | INFO | WARNING | ERROR
wan_password_hash = ""            # sha256:<hex> — run: egovault web-password

[llm]
provider          = "llama_cpp"   # "llama_cpp" | "openai" | any OpenAI-compatible
model             = "gemma-4-e2b-it"
base_url          = "http://127.0.0.1:8080"
api_key           = ""            # required only for cloud providers
timeout_seconds   = 300
chunk_target_tokens = 2000

[llama_cpp]
manage          = true
model_path      = "./models/gemma-4-E2B-it-UD-Q4_K_XL.gguf"
model_hf_repo   = "unsloth/gemma-4-E2B-it-GGUF"
flash_attn      = true
vram_budget_pct = 0.80
n_gpu_layers    = 99
ctx_size        = 0               # 0 = auto-compute from free VRAM

[reranker]
enabled = true
backend = "auto"                  # "bm25" | "cross-encoder" | "auto"
model   = "cross-encoder/ms-marco-MiniLM-L-6-v2"

[embeddings]
enabled            = true
provider           = ""           # "" = inherit [llm] provider
model              = "nomic-embed-text"
base_url           = ""           # "" = inherit [llm] base_url
hyde_enabled       = true
contextual_enabled = true
hype_enabled       = true

[crag]
enabled   = true
threshold = 0.10
strategy  = "hyde"                # "hyde" | "broaden" | "empty"

[sentence_window]
enabled     = true
window_size = 3
overlap     = 1

[scheduler]
auto_refresh_inbox_minutes    = 30
auto_refresh_gmail_minutes    = 30
auto_refresh_telegram_minutes = 30

[web_search]
provider     = "duckduckgo"       # "duckduckgo" | "searxng" | "" (disabled)
max_results  = 5
searxng_url  = ""                 # e.g. "http://127.0.0.1:8888"
fallback_urls = []
categories   = "general"

[adapters]

[adapters.facebook]
encoding_fix = true

[adapters.chromium]
profile_path     = ""
copy_before_read = true

[telegram]
token            = ""             # set your bot token, or export EGOVAULT_TELEGRAM_TOKEN
allowed_chat_ids = []             # your Telegram chat ID(s) — find with @userinfobot
top_n            = 10
"""


def _bootstrap_personal_config() -> None:
    """Write data/egovault.toml with defaults on first run."""
    if _PERSONAL_CONFIG.exists():
        return
    _PERSONAL_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    _PERSONAL_CONFIG.write_text(_DEFAULT_CONFIG_CONTENT, encoding="utf-8")
    logger.info(
        "Created personal config at %s — edit it to set your credentials.",
        _PERSONAL_CONFIG,
    )


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from the first config file found, or return defaults."""
    global _settings

    if config_path is None:
        _bootstrap_personal_config()

    paths = [config_path] if config_path else _DEFAULT_CONFIG_PATHS
    raw: dict = {}

    for path in paths:
        if path and path.exists():
            with open(path, "rb") as f:
                raw = tomllib.load(f)
            logger.debug("Loaded config from %s", path)
            break

    general = raw.get("general", {})
    # Accept both [llm] (new) and legacy [ollama] section names.
    llm_raw = raw.get("llm", raw.get("ollama", {}))

    llm = LLMSettings(
        provider=llm_raw.get("provider", "llama_cpp"),
        model=llm_raw.get("model", "gemma-4-e4b-it"),
        base_url=llm_raw.get("base_url", "http://127.0.0.1:8080"),
        api_key=llm_raw.get("api_key", ""),
        timeout_seconds=llm_raw.get("timeout_seconds", 300),
        chunk_target_tokens=llm_raw.get("chunk_target_tokens", 2000),
        max_ctx_tokens=int(llm_raw.get("max_ctx_tokens", 0)),
    )

    reranker_raw = raw.get("reranker", {})
    reranker = RerankerSettings(
        enabled=reranker_raw.get("enabled", True),
        backend=reranker_raw.get("backend", "auto"),
        model=reranker_raw.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    )

    embed_raw = raw.get("embeddings", {})
    embeddings = EmbeddingSettings(
        enabled=embed_raw.get("enabled", False),
        provider=embed_raw.get("provider", ""),
        model=embed_raw.get("model", "nomic-embed-text"),
        base_url=embed_raw.get("base_url", ""),
        hyde_enabled=embed_raw.get("hyde_enabled", True),
        contextual_enabled=embed_raw.get("contextual_enabled", True),
        hype_enabled=embed_raw.get("hype_enabled", True),
    )

    crag_raw = raw.get("crag", {})
    crag = CRAGSettings(
        enabled=crag_raw.get("enabled", True),
        threshold=float(crag_raw.get("threshold", 0.10)),
        strategy=crag_raw.get("strategy", "hyde"),
    )

    sw_raw = raw.get("sentence_window", {})
    sentence_window = SentenceWindowSettings(
        enabled=sw_raw.get("enabled", True),
        window_size=int(sw_raw.get("window_size", 3)),
        overlap=int(sw_raw.get("overlap", 1)),
    )

    gmail_raw = raw.get("gmail", {})
    gmail = GmailOAuthSettings(
        client_id=(
            os.environ.get("EGOVAULT_GMAIL_CLIENT_ID")
            or gmail_raw.get("client_id", "")
        ),
        client_secret=(
            os.environ.get("EGOVAULT_GMAIL_CLIENT_SECRET")
            or gmail_raw.get("client_secret", "")
        ),
    )

    lcpp_raw = raw.get("llama_cpp", {})
    llama_cpp = LlamaCppSettings(
        base_url=lcpp_raw.get("base_url", "http://127.0.0.1:8080"),
        n_gpu_layers=int(lcpp_raw.get("n_gpu_layers", 99)),
        ctx_size=int(lcpp_raw.get("ctx_size", 0)),
        embed=bool(lcpp_raw.get("embed", True)),
        manage=bool(lcpp_raw.get("manage", True)),
        model_path=str(lcpp_raw.get("model_path", "./models/gemma-4-E2B-it-UD-Q4_K_XL.gguf")),
        flash_attn=bool(lcpp_raw.get("flash_attn", True)),
        vram_budget_pct=float(lcpp_raw.get("vram_budget_pct", 0.80)),
        model_hf_repo=str(lcpp_raw.get("model_hf_repo", "unsloth/gemma-4-E2B-it-GGUF")),
    )

    ws_raw = raw.get("web_search", {})
    web_search = WebSearchSettings(
        provider=ws_raw.get("provider", "duckduckgo"),
        searxng_url=ws_raw.get("searxng_url", ""),
        fallback_urls=list(ws_raw.get("fallback_urls", [])),
        max_results=int(ws_raw.get("max_results", 5)),
        categories=ws_raw.get("categories", "general"),
    )

    tg_raw = raw.get("telegram", {})
    telegram = TelegramSettings(
        token=(os.environ.get("EGOVAULT_TELEGRAM_TOKEN") or tg_raw.get("token", "")),  # noqa: SIM112
        allowed_chat_ids=[int(x) for x in tg_raw.get("allowed_chat_ids", [])],
        top_n=int(tg_raw.get("top_n", 10)),
    )

    _settings = Settings(
        vault_db=general.get("vault_db", "./data/vault.db"),
        output_dir=general.get("output_dir", "./output"),
        inbox_dir=general.get("inbox_dir", "./inbox"),
        log_level=general.get("log_level", "INFO"),
        wan_password_hash=_resolve_wan_password_hash(
            general.get("wan_password_hash", ""),
            vault_db=general.get("vault_db", "./data/vault.db"),
        ),
        llm=llm,
        llama_cpp=llama_cpp,
        reranker=reranker,
        embeddings=embeddings,
        crag=crag,
        sentence_window=sentence_window,
        gmail=gmail,
        scheduler=_load_scheduler_settings(raw),
        web_search=web_search,
        telegram=telegram,
    )
    return _settings


def _resolve_wan_password_hash(toml_value: str, vault_db: str = "./data/vault.db") -> str:
    """Return wan_password_hash from toml or, if empty, from data/wan.password."""
    if toml_value:
        return toml_value
    pw_file = Path(vault_db).parent / "wan.password"
    if pw_file.exists():
        return pw_file.read_text(encoding="utf-8").strip()
    return ""


def _load_scheduler_settings(raw: dict) -> "SchedulerSettings":
    s = raw.get("scheduler", {})
    return SchedulerSettings(
        auto_refresh_inbox_minutes=int(s.get("auto_refresh_inbox_minutes", 0)),
        auto_refresh_gmail_minutes=int(s.get("auto_refresh_gmail_minutes", 0)),
        auto_refresh_telegram_minutes=int(s.get("auto_refresh_telegram_minutes", 0)),
    )


def get_settings() -> Settings:
    """Return cached settings, loading defaults if not yet loaded."""
    if _settings is None:
        return load_settings()
    return _settings


def reset_settings() -> None:
    """Clear the cached settings singleton — useful for test isolation."""
    global _settings
    _settings = None
    load_agent_prompts.cache_clear()


def configure_logging(settings: Settings) -> None:
    """Configure the root logger from settings."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


_DEFAULT_ENRICH_PROMPT = (
    "You are a personal knowledge extractor. Given a batch of messages or document text, "
    "extract only actionable insights, shared links, and meaningful updates. "
    "Ignore greetings, filler, and logistics noise.\n\n"
    "Output format (strict):\n"
    "SUMMARY: <one paragraph>\n"
    "GEMS:\n"
    "- [Link] <url> — <who shared it, when>\n"
    "- [Decision] <what was decided>\n"
    "- [Recommendation] <what was recommended>\n"
    "- [Action] <action item and who owns it>"
)

_DEFAULT_CHAT_PROMPT = (
    "You are the user's personal AI ego — a sharp, candid second brain and general-purpose assistant. "
    "When vault context is provided (marked [VAULT CONTEXT]), draw on it to give personalised, "
    "grounded answers and cite the source (platform and date) when relevant. "
    "When no vault context is provided, answer from your own knowledge as a knowledgeable assistant — "
    "never refuse a general question. "
    "Keep replies concise and conversational.\n\n"
    "You are aware of the following chat commands that the user can run at any time:\n"
    "  /scan <folder>  — scan a folder once and add its files to the vault so you can answer questions about them.\n"
    "                    <folder> can be a well-known alias (desktop, documents, downloads, pictures, music, videos, movies, home)\n"
    "                    or any absolute or tilde path (e.g. ~/notes or C:\\work\\docs).\n"
    "  /scan --list    — show all well-known folder aliases and their resolved paths for this system.\n"
    "  /sources        — show which vault records informed the last answer.\n"
    "  /top N          — change how many vault records are retrieved per query (default 10, max 50).\n"
    "  /status         — show the loaded LLM model, VRAM usage, and GPU layer count.\n"
    "  /clear          — clear the terminal screen.\n"
    "  /restart        — clear the screen and reset the conversation history for this session.\n"
    "  /exit or /quit  — end the chat session.\n"
    "  /help           — show the full command list.\n\n"
    "Natural language also works — you will recognise phrases like:\n"
    "  \"scan my downloads folder\"  →  /scan downloads\n"
    "  \"index my desktop\"          →  /scan desktop\n"
    "  \"show sources\"              →  /sources\n"
    "  \"please quit\"               →  /exit\n"
    "  \"clear the screen\"          →  /clear\n"
    "  \"start over\"                →  /restart\n\n"
    "If the user asks you to scan a folder, look at their files, import documents, or index a directory, "
    "remind them they can use /scan <folder> to do that instantly without leaving the chat."
)

_DEFAULT_WELCOME_TEXT = (
    "Hello! I'm your EgoVault — a personal second brain that knows your notes, conversations, "
    "and saved content. Ask me anything, or type /help to see available commands."
)


_AGENT_MD_TEMPLATE = (
    "## enrichment\n"
    "{enrichment}\n"
    "\n"
    "## chat\n"
    "{chat}\n"
    "\n"
    "## welcome\n"
    "{welcome}\n"
)


@lru_cache(maxsize=4)
def load_agent_prompts(path: Path | None = None) -> dict[str, str]:
    """Load agent system prompts from data/agent.md, falling back to built-in defaults.

    The file uses ## headings as section names ('enrichment', 'chat').
    If the file does not exist it is created from the built-in defaults.
    Missing sections fall back to the built-in defaults.
    """
    result: dict[str, str] = {"enrichment": _DEFAULT_ENRICH_PROMPT, "chat": _DEFAULT_CHAT_PROMPT, "welcome": _DEFAULT_WELCOME_TEXT}
    agent_path = path or Path("./data/agent.md")
    if not agent_path.exists():
        try:
            agent_path.parent.mkdir(parents=True, exist_ok=True)
            agent_path.write_text(
                _AGENT_MD_TEMPLATE.format(
                    enrichment=_DEFAULT_ENRICH_PROMPT,
                    chat=_DEFAULT_CHAT_PROMPT,
                    welcome=_DEFAULT_WELCOME_TEXT,
                ),
                encoding="utf-8",
            )
            logger.info("Created default agent prompts file at %s", agent_path)
        except OSError as exc:
            logger.warning("Could not create %s: %s", agent_path, exc)
        return result

    text = agent_path.read_text(encoding="utf-8")
    current_key: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            if current_key is not None:
                result[current_key] = "\n".join(current_lines).strip()
            current_key = line[3:].strip().lower()
            current_lines = []
        elif current_key is not None:
            current_lines.append(line)

    if current_key is not None:
        result[current_key] = "\n".join(current_lines).strip()

    return result

