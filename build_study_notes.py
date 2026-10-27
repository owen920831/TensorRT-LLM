# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the offline study reader from the canonical Markdown notes.

Run: uv run --no-project --with markdown python build_study_notes.py
"""

import html
import re
from pathlib import Path

import markdown


def _render_markdown(text: str) -> str:
    rendered = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    return re.sub(
        r"(<table>.*?</table>)", r'<div class="table-wrap">\1</div>', rendered, flags=re.S
    )


def _build_page(root: Path) -> str:
    source = (root / "compiler-and-flashinfer-e2e.md").read_text(encoding="utf-8")
    parts = re.split(r"^## (\d{2})｜(.+)$", source, flags=re.M)
    chapters = []
    navigation = []
    groups = {
        "00": "建立地圖 · 00—03",
        "04": "13 集主題 · 04—16",
        "17": "FlashInfer 與接縫 · 17—21",
        "22": "驗證、短課與來源 · 22—25",
        "26": "回到完整模型 · 26—28",
    }
    for offset in range(1, len(parts), 3):
        number, title, body = parts[offset : offset + 3]
        identifier = f"chapter-{number}"
        safe_title = html.escape(title)
        if number in groups:
            navigation.append(f'<div class="group">{groups[number]}</div>')
        navigation.append(f'<a href="#{identifier}">{number} · {safe_title}</a>')
        next_button = ""
        if offset + 3 < len(parts):
            next_number = parts[offset + 3]
            next_button = f'<a href="#chapter-{next_number}">下一節 →</a>'
        chapters.append(
            f'<details class="chapter" id="{identifier}">'
            f'<summary><span class="num">{number}</span>{safe_title}</summary>'
            f'<div class="chapter-body">{_render_markdown(body)}'
            f'<div class="chapter-foot"><button data-bookmark="{identifier}">記住這一節</button>'
            f"{next_button}</div></div></details>"
        )
    if not chapters:
        raise ValueError("No numbered lesson sections found in the Markdown source")
    ledger = (root / "bilibili-compiler-sources.md").read_text(encoding="utf-8")
    template = (root / "e2e-learning.template.html").read_text(encoding="utf-8")
    page = template.replace("@@NAV@@", "\n".join(navigation))
    page = page.replace("@@CHAPTERS@@", "\n".join(chapters))
    page = page.replace("@@COUNT@@", str(len(chapters)))
    return page.replace("@@SOURCES@@", _render_markdown(ledger))


if __name__ == "__main__":
    _root = Path(__file__).resolve().parent
    _output = _root / "e2e-learning.html"
    _output.write_text(_build_page(_root), encoding="utf-8")
    print(f"Built {_output.name}")
