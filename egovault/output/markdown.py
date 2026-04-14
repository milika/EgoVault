"""Markdown output generator — produces YAML-frontmatter .md files from enriched records."""
from __future__ import annotations

import re
from pathlib import Path

from egovault.config import Settings
from egovault.core.schema import NormalizedRecord
from egovault.core.store import VaultStore


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _get_enrichment(store: VaultStore, record_id: str) -> dict | None:
    cur = store._con.execute(
        "SELECT summary, gems_raw FROM enrichment_results WHERE record_id = ?",
        (record_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return {"summary": row["summary"], "gems_raw": row["gems_raw"]}


def _get_gems(store: VaultStore, record_id: str) -> list[dict]:
    cur = store._con.execute(
        "SELECT gem_type, content FROM extracted_gems WHERE record_id = ? ORDER BY id",
        (record_id,),
    )
    return [{"gem_type": r["gem_type"], "content": r["content"]} for r in cur.fetchall()]


def _gem_label(gem_type: str) -> str:
    return gem_type.capitalize()


def generate_markdown(
    record: NormalizedRecord,
    store: VaultStore,
    output_dir: Path,
) -> Path | None:
    """Generate a Markdown file for *record* and return its path.

    Returns None if the record has no enrichment result.
    """
    enrichment = _get_enrichment(store, record.id)
    if enrichment is None:
        return None

    gems = _get_gems(store, record.id)

    # --- output path ---
    ts = record.timestamp
    date_str = f"{ts.year:04d}-{ts.month:02d}"
    source_name = _slug(record.thread_name or record.file_path or record.platform)
    filename = f"{record.platform}_{source_name}_{date_str}.md"
    out_path = output_dir / filename

    # --- frontmatter ---
    participants: list[str] = []
    if record.sender_name and record.sender_name != "user":
        participants.append(record.sender_name)

    frontmatter_lines = [
        "---",
        f"platform: {record.platform}",
        f'source: "{record.thread_name or record.file_path or ""}"',
        f"participants: {participants}",
        f'date_range: "{date_str}"',
        f"tags: [egovault, {record.platform}]",
        "private: false",
        "shareable: false",
        "---",
    ]

    # --- summary section ---
    summary_section = ["", "## Summary", "", enrichment["summary"] or "_No summary generated._"]

    # --- gems section ---
    gems_section = ["", "## Gems"]
    if gems:
        for gem in gems:
            gems_section.append(f"- [{_gem_label(gem['gem_type'])}] {gem['content']}")
    else:
        gems_section.append("_No gems extracted._")

    # --- raw context section ---
    raw_section = [
        "",
        "## Raw Context",
        "",
        "<details>",
        "<summary>Original content</summary>",
        "",
        record.body,
        "",
        "</details>",
    ]

    content = "\n".join(frontmatter_lines + summary_section + gems_section + raw_section) + "\n"

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


class MarkdownGenerator:
    def __init__(self, store: VaultStore, settings: Settings) -> None:
        self._store = store
        self._output_dir = Path(settings.output_dir)

    def generate(self, record: NormalizedRecord) -> Path | None:
        return generate_markdown(record, self._store, self._output_dir)

    def generate_all(self) -> list[Path]:
        """Generate Markdown for all enriched records (enriched = 1)."""
        records = self._store.get_records({"enriched": 1})
        paths: list[Path] = []
        for rec in records:
            p = self.generate(rec)
            if p:
                paths.append(p)
        return paths
