"""EgoVault — Hugging Face Spaces demo.

Synthetic data only.  LLM calls go to HF Inference API (Mistral-7B).
EgoVault is designed to run 100% locally on your machine via llama-server.
Get the real thing: https://github.com/milika/egovault
"""
from __future__ import annotations

import os
import sqlite3
import textwrap
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EgoVault · Demo",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Demo banner
# ---------------------------------------------------------------------------

st.info(
    "**Demo mode** — synthetic data, LLM via HF Inference API.  "
    "EgoVault is designed to run **100 % locally** on your machine.  "
    "[Get the real thing on GitHub →](https://github.com/milika/egovault)",
    icon="ℹ️",
)

# ---------------------------------------------------------------------------
# Synthetic records
# ---------------------------------------------------------------------------

_RECORDS = [
    # (id, platform, record_type, timestamp, sender_name, thread_name, body)
    ("r01", "gmail", "email", "2025-11-01T09:15:00Z", "Alice Johnson", "Project Nexus",
     "Hi, quick reminder — the Q4 design review is Friday at 2 pm. Please bring the latest "
     "mockups. The client specifically asked about the dark mode implementation."),
    ("r02", "gmail", "email", "2025-11-02T14:30:00Z", "Bob Martinez", "Project Nexus",
     "Attached the updated wireframes for the dashboard. Dark mode toggle added as discussed. "
     "Color palette follows brand guidelines. Let me know if we need to revise the typography."),
    ("r03", "gmail", "email", "2025-11-03T08:00:00Z", "Carol Lin", "Team Updates",
     "The Barcelona team confirmed the meeting for December 5th. They want to discuss the "
     "migration plan to the new architecture. Flights booked — arriving the 4th evening."),
    ("r04", "gmail", "email", "2025-11-10T11:45:00Z", "Alice Johnson", "Budget 2026",
     "Please review the attached budget proposal for 2026. Key items: infrastructure costs up "
     "15 % due to the new GPU cluster, R&D allocation increased to 30 %, travel budget cut 20 %."),
    ("r05", "gmail", "email", "2025-12-01T16:20:00Z", "DevOps Team", "Incident Report",
     "Post-mortem for the Nov 28 outage: root cause was a memory leak in the embedding service. "
     "Fix deployed at 03:40 UTC. Action items: add memory monitoring, stress-test the embeddings pipeline."),
    ("r06", "telegram", "message", "2025-11-05T20:10:00Z", "Marco", "Friends Group",
     "Anyone up for hiking next weekend? Thinking the trail behind the old mill — about 12 km, "
     "moderate difficulty. Bring layers, weather should be chilly."),
    ("r07", "telegram", "message", "2025-11-06T09:30:00Z", "Sara", "Friends Group",
     "I'm in! Also — has anyone tried the new coffee place on Graben street? Ridiculously good "
     "espresso. We should go before the hike."),
    ("r08", "telegram", "message", "2025-11-15T18:00:00Z", "Marco", "Friends Group",
     "Reminder: my sister's birthday dinner is Saturday 22nd at 7 pm, Casa Nostra restaurant. "
     "Please RSVP by Thursday so I can confirm the reservation."),
    ("r09", "telegram", "message", "2025-12-03T12:45:00Z", "Sara", "2026 Plans",
     "For the New Year trip — found great deals on flights to Lisbon. Fly out Dec 29, return Jan 3. "
     "The Airbnb in Alfama sleeps 6, very reasonable split 3 ways."),
    ("r10", "local_inbox", "file", "2025-10-20T00:00:00Z", "self", "Notes",
     "## Ideas for EgoVault\n- Add WhatsApp export support\n- Build a timeline view by date\n"
     "- Investigate DuckDB for analytics queries\n- iOS Shortcut to capture quick notes into the vault"),
    ("r11", "local_inbox", "file", "2025-11-12T00:00:00Z", "self", "Reading List",
     "Papers to read:\n- RAG for Knowledge-Intensive NLP Tasks (Lewis et al.)\n"
     "- Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE paper)\n"
     "- Contextual Retrieval blog post (Anthropic)\n\n"
     "Books: The Pragmatic Programmer (ch 8), Designing Data-Intensive Applications (reread ch 3)"),
    ("r12", "local_inbox", "file", "2025-12-10T00:00:00Z", "self", "Recipe: Pasta e Fagioli",
     "Ingredients: 400g borlotti beans, 200g ditaloni pasta, 1 onion, 2 carrots, celery, "
     "pancetta, rosemary, olive oil, parmesan.\n"
     "Method: Sauté vegetables and pancetta. Add beans, stock, rosemary. Simmer 20 min. "
     "Blend half, add pasta, cook al dente. Finish with olive oil and parmesan."),
]

# ---------------------------------------------------------------------------
# DB init
# ---------------------------------------------------------------------------

_DB_PATH = Path("data/demo_vault.db")
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS normalized_records (
    id          TEXT PRIMARY KEY,
    platform    TEXT NOT NULL,
    record_type TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    sender_name TEXT NOT NULL DEFAULT '',
    thread_name TEXT NOT NULL DEFAULT '',
    body        TEXT NOT NULL DEFAULT ''
);

CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(
    body, thread_name, sender_name,
    content='normalized_records',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS records_ai AFTER INSERT ON normalized_records BEGIN
    INSERT INTO records_fts(rowid, body, thread_name, sender_name)
    VALUES (new.rowid, new.body, new.thread_name, new.sender_name);
END;
"""


@st.cache_resource
def _get_db() -> sqlite3.Connection:
    con = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(_DDL)
    if con.execute("SELECT COUNT(*) FROM normalized_records").fetchone()[0] == 0:
        con.executemany(
            "INSERT OR IGNORE INTO normalized_records "
            "(id, platform, record_type, timestamp, sender_name, thread_name, body) "
            "VALUES (?,?,?,?,?,?,?)",
            _RECORDS,
        )
        con.commit()
    return con


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _search(query: str, top_n: int) -> list[sqlite3.Row]:
    con = _get_db()
    try:
        rows = con.execute(
            """
            SELECT r.id, r.platform, r.record_type, r.timestamp,
                   r.sender_name, r.thread_name, r.body
            FROM records_fts fts
            JOIN normalized_records r ON r.rowid = fts.rowid
            WHERE records_fts MATCH ?
            ORDER BY fts.rank
            LIMIT ?
            """,
            (query, top_n),
        ).fetchall()
    except Exception:
        rows = []
    if not rows:
        like = f"%{query}%"
        rows = con.execute(
            "SELECT id, platform, record_type, timestamp, sender_name, thread_name, body "
            "FROM normalized_records "
            "WHERE body LIKE ? OR thread_name LIKE ? OR sender_name LIKE ? "
            "LIMIT ?",
            (like, like, like, top_n),
        ).fetchall()
    return rows


def _format_context(rows: list[sqlite3.Row]) -> str:
    parts = []
    for r in rows:
        parts.append(
            f"[{r['platform']} · {r['record_type']} · {r['timestamp'][:10]}]\n"
            f"From: {r['sender_name']}   Thread: {r['thread_name']}\n"
            f"{r['body']}"
        )
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM — HF Inference API
# ---------------------------------------------------------------------------

_MODEL = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
_HF_TOKEN = os.environ.get("HF_TOKEN") or None


def _chat_stream(user_msg: str, history: list[dict], rows: list[sqlite3.Row]):
    from huggingface_hub import InferenceClient

    client = InferenceClient(model=_MODEL, token=_HF_TOKEN)
    system = textwrap.dedent(f"""
        You are a personal AI assistant with read access to the user's private data vault.
        Use only the retrieved records below to answer.  If they don't contain enough
        information, say so honestly.  Be concise.

        --- RETRIEVED RECORDS ---
        {_format_context(rows)}
        --- END RECORDS ---
    """).strip()

    messages = [{"role": "system", "content": system}]
    for h in history[-6:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_msg})

    for chunk in client.chat_completion(messages=messages, max_tokens=512, stream=True):
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🔒 EgoVault")
    st.caption("Local-first personal data vault · Demo")
    st.divider()

    top_n = st.slider("Records to retrieve", 1, 10, 5)

    con = _get_db()
    total = con.execute("SELECT COUNT(*) FROM normalized_records").fetchone()[0]
    platforms = con.execute(
        "SELECT platform, COUNT(*) c FROM normalized_records GROUP BY platform ORDER BY c DESC"
    ).fetchall()
    st.metric("Vault records (demo)", total)
    for p in platforms:
        st.caption(f"  {p['platform']}: {p['c']}")

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conv_history = []
        st.rerun()

    st.divider()
    st.caption(f"⚙️ Model: `{_MODEL.split('/')[-1]}`")
    st.caption("[Get the local version →](https://github.com/milika/egovault)")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_WELCOME = (
    "Hi! I'm **EgoVault** (demo mode).  I have 12 synthetic records across "
    "Gmail, Telegram, and local files.  Try:\n\n"
    "- *What's happening with the Barcelona trip?*\n"
    "- *Any ideas about EgoVault features?*\n"
    "- *When is Marco's sister's birthday dinner?*\n"
    "- *What's the 2026 budget situation?*"
)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": _WELCOME}]

if "conv_history" not in st.session_state:
    st.session_state.conv_history = []

# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------

st.title("EgoVault")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your vault…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    results = _search(prompt, top_n)
    answer = ""

    with st.chat_message("assistant"):
        if not results:
            answer = "I couldn't find any relevant records for that query in the demo vault."
            st.markdown(answer)
        else:
            placeholder = st.empty()
            try:
                for chunk in _chat_stream(prompt, st.session_state.conv_history, results):
                    answer += chunk
                    placeholder.markdown(answer + "▌")
                placeholder.markdown(answer)
            except Exception as exc:
                answer = (
                    f"⚠️ LLM unavailable ({exc}).\n\n"
                    f"**Raw retrieved records:**\n\n{_format_context(results)}"
                )
                placeholder.markdown(answer)

        if results:
            with st.expander(f"📄 {len(results)} source(s) retrieved"):
                for r in results:
                    st.markdown(
                        f"**{r['platform']} · {r['thread_name']}**  "
                        f"({r['timestamp'][:10]})"
                    )
                    body_preview = r["body"][:200] + ("…" if len(r["body"]) > 200 else "")
                    st.caption(body_preview)
                    st.divider()

    st.session_state.conv_history.extend([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ])
    st.session_state.messages.append({"role": "assistant", "content": answer})
