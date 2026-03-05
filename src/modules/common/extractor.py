"""
Generic extractor for many news/blog sites.

Public API:
- Extractor(html): parse HTML once
- extract_title() -> Optional[str]
- extract_description() -> Optional[str]
- extract_content(fit_markdown: str) -> str   # clean FitMarkdown (no HTML needed)

Design goals:
- High generality across domains
- Deterministic, testable heuristics
- Minimal, high-confidence keyword lists

This variant: content output drops ALL headings and short/noisy lines,
keeping only main body paragraphs (high precision).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional, List, Iterable, Any

from bs4 import BeautifulSoup


_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S+")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


# -----------------------------
# Utilities
# -----------------------------

def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _norm_key(s: str) -> str:
    return _norm_space(s).lower()


def _strip_control_chars(s: str) -> str:
    # keep normal unicode text; remove low ASCII controls
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", s or "")


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[:n].rstrip()


def _looks_like_sentence(text: str) -> bool:
    # Lightweight: presence of sentence punctuation OR CJK punctuation
    return bool(re.search(r"[.!?。！？]", text))


def _is_heading(line: str) -> bool:
    return bool(_HEADING_RE.match(line.strip()))


def _remove_title_suffix(title: str) -> str:
    """
    Remove common site suffix patterns:
    - "Title | Site"
    - "Title - Site"
    - "Title — Site"
    Keep the left part when it looks like a proper title.
    """
    t = _norm_space(title)
    if not t:
        return t

    parts = re.split(r"\s+[\|\-—–:]\s+", t)
    if len(parts) <= 1:
        return t

    left = parts[0].strip()
    if len(left) < 8:
        return t
    return left


# -----------------------------
# HTML Extractor
# -----------------------------

@dataclass(frozen=True)
class _Candidate:
    source: str
    text: str


class Extractor:
    """
    Parse HTML once and provide:
    - extract_title()
    - extract_description()
    - extract_content(fit_markdown)

    Note: Content cleaning is markdown-only (FitMarkdown already pre-cleaned).
    """

    # Minimal, high-confidence noise hints for *titles*
    _TITLE_NOISE_HINTS = (
        "subscribe",
        "newsletter",
        "sign in",
        "log in",
        "cookie",
        "privacy",
        "terms",
        "advertisement",
    )

    # Caption / player UI tokens (high precision)
    _UI_TOKENS = (
        "video player is loading",
        "this is a modal window",
        "beginning of dialog window",
        "end of dialog window",
        "escape will cancel and close the window",
        "text color",
        "text background color",
        "caption",
        "caption area",
        "background color",
        "opacity",
        "font size",
        "font family",
        "text edge",
        "edge style",
        "drop shadow",
        "raised",
        "depressed",
        "uniform",
        "proportional",
        "sans-serif",
        "monospace",
        "serif",
        "semi-transparent",
        "transparent",
        "opaque",
        # colors (they appear in UI menus)
        "white",
        "black",
        "red",
        "green",
        "blue",
        "yellow",
        "magenta",
        "cyan",
    )

    # Glue pattern for "Text ColorWhite" / "OpacityOpaque" style concatenation
    _UI_GLUE_PAT = re.compile(
        r"(Text\s*Color|Opacity|Font\s*Size|Font\s*Family|Text\s*Background\s*Color|"
        r"Caption\s*Area\s*Background\s*Color)"
        r"(White|Black|Red|Green|Blue|Yellow|Magenta|Cyan|Opaque|Transparent|Semi-Transparent|\d{1,3}%|\w+)",
        flags=re.I,
    )

    # Inline noise hints (keep small, high precision)
    _INLINE_NOISE_HINTS = (
        "sign up",
        "sign in",
        "log in",
        "create account",
        "subscribe",
        "newsletter",
        "unlock unlimited access",
        "privacy policy",
        "terms of service",
        "cookie policy",
        "all rights reserved",
        "copyright",
        "advertisement",
        "sponsored",
        "promo code",
        "use code",
        "watch live",
        "live:",
        "listen",
        "share",
        "copy link",
        "print",
        "report",
        "support the investigative reporting",
        "vip access",
    )

    _SEE_ALL_PAT = re.compile(r"^(see|view|show)\s+(all|more)\b", flags=re.I)

    # -----------------------------
    # Content selection knobs
    # -----------------------------
    # 行/段落太短通常是频道/推荐/按钮/面包屑。你要“只保留主要内容”，这里要更激进。
    _MIN_PARAGRAPH_CHARS = 60           # 句子/段落最小长度（可按站点调：50~120）
    _MIN_PARAGRAPH_WORDLIKE_TOKENS = 10 # 句子/段落最小 token 数（中英通用，粗略）
    _MIN_SENTENCE_PUNCT_RATIO = 0.0     # 不强制比例，只要有句号类标点即可（见 _looks_like_sentence）

    def __init__(self, html: str):
        self.html = html or ""
        self.soup = BeautifulSoup(self.html, "lxml") if self.html else BeautifulSoup("", "lxml")

    # -----------------------------
    # Public: Title / Description
    # -----------------------------

    def extract_title(self) -> Optional[str]:
        """
        Generic title extraction:
        1) <h1>
        2) meta keys that look like title/headline
        3) JSON-LD headline/name
        4) <title> (with suffix removal)
        Then select best candidate by validity rules.
        """
        candidates: List[_Candidate] = []

        h1 = self.soup.find("h1")
        if h1:
            t = _norm_space(h1.get_text(" ", strip=True))
            if t:
                candidates.append(_Candidate("h1", t))

        meta_map = self._meta_map()
        for k, v in meta_map.items():
            if "title" in k or "headline" in k or k.endswith("dc.title") or k.endswith("dcterms.title"):
                if v:
                    candidates.append(_Candidate(f"meta:{k}", v))

        jl = self._extract_json_ld_first(("headline", "name", "title"))
        if jl:
            candidates.append(_Candidate("jsonld", jl))

        if self.soup.title and self.soup.title.text:
            tt = _norm_space(self.soup.title.text)
            tt = _remove_title_suffix(tt)
            if tt:
                candidates.append(_Candidate("title_tag", tt))

        return self._pick_best_title(candidates)

    def extract_description(self) -> Optional[str]:
        """
        Generic description extraction:
        1) meta keys that look like description/summary/abstract
        2) JSON-LD description/abstract
        3) first paragraph text (from likely main containers), truncated to 240 chars
        """
        meta_map = self._meta_map()
        desc_candidates: List[_Candidate] = []
        for k, v in meta_map.items():
            if not v:
                continue
            if (
                "description" in k
                or "summary" in k
                or "abstract" in k
                or k.endswith("dc.description")
                or k.endswith("dcterms.description")
            ):
                desc_candidates.append(_Candidate(f"meta:{k}", v))

        best_meta = self._pick_best_description([c.text for c in desc_candidates])
        if best_meta:
            return best_meta

        jl = self._extract_json_ld_first(("description", "abstract"))
        if jl:
            jl = self._clean_inline_text(jl)
            if jl:
                return _truncate(jl, 240) or None

        p = self._extract_first_paragraph_text()
        if p:
            p = self._clean_inline_text(p)
            p = _truncate(p, 240)
            return p or None

        return None

    # -----------------------------
    # Public: Content cleaning (FitMarkdown)
    # -----------------------------

    def extract_content(self, fit_markdown: str) -> str:
        """
        High-precision main content extraction from FitMarkdown.

        Output characteristics:
        - NO headings at all
        - NO short lines/paragraphs
        - NO UI/nav/related/subscription blocks
        - Only substantial body paragraphs (sentence-like, sufficiently long)
        """
        return self.clean_fit_markdown(fit_markdown)

    def clean_fit_markdown(self, markdown: str) -> str:
        if not markdown or not markdown.strip():
            return ""

        md = markdown.replace("\r\n", "\n").replace("\r", "\n")

        # 1) 行级过滤：去 heading / 去 code block / 去 UI / 去短噪声
        filtered_lines = self._filter_lines_strict(md)
        if not filtered_lines:
            return ""

        # 2) 行级筛选：正文段落 + markdown 表格行
        lines: List[str] = []
        for idx, ln in enumerate(filtered_lines):
            stripped = ln.strip()
            if not stripped:
                continue
            if self._is_markdown_table_line(filtered_lines, idx):
                lines.append(stripped)
                continue
            if self._is_body_paragraph(stripped):
                lines.append(stripped)
        if not lines:
            return ""

        # 3) 最终输出：每行一行（纯正文）
        return "\n".join(lines).strip()

    # -----------------------------
    # Internals: HTML
    # -----------------------------

    def _meta_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for tag in self.soup.find_all("meta"):
            key = tag.get("name") or tag.get("property") or tag.get("itemprop") or tag.get("http-equiv")
            val = tag.get("content")
            if not key or not val:
                continue
            k = _norm_key(str(key))
            v = _clean_meta_text(str(val))
            if not k or not v:
                continue
            if k not in out:
                out[k] = v
        return out

    def _extract_json_ld_first(self, keys: tuple[str, ...]) -> Optional[str]:
        for script in self.soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
            raw = script.string
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            for item in self._iter_json_ld(payload):
                if not isinstance(item, dict):
                    continue
                for k in keys:
                    val = item.get(k)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
        return None

    @staticmethod
    def _iter_json_ld(payload: Any) -> Iterable[Any]:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            graph = payload.get("@graph")
            if isinstance(graph, list):
                return graph
            return [payload]
        return []

    def _extract_first_paragraph_text(self) -> Optional[str]:
        soup = self.soup

        containers = []
        for sel in ("article", "main"):
            containers.extend(soup.find_all(sel))
        containers.extend(soup.find_all(attrs={"role": "main"}))

        if not containers:
            containers = [soup.body] if soup.body else [soup]

        for c in containers:
            for p in c.find_all("p"):
                txt = _norm_space(p.get_text(" ", strip=True))
                txt = _strip_control_chars(txt)
                if len(txt) >= 80:
                    return txt
        return None

    def _pick_best_title(self, candidates: List[_Candidate]) -> Optional[str]:
        scored: List[tuple[float, str]] = []
        for c in candidates:
            t = _strip_control_chars(_norm_space(c.text))
            t = _remove_title_suffix(t)
            if not self._is_valid_title(t):
                continue
            score = 0.0
            if c.source == "h1":
                score += 2.0
            if len(t) > 120:
                score -= 2.0
            elif len(t) > 90:
                score -= 1.0
            score += min(1.5, (len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", t)) / max(1, len(t))) * 1.5)
            score -= t.count("|") * 0.7 + t.count(" - ") * 0.5
            scored.append((score, t))

        if not scored:
            return None
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1].strip() or None

    def _pick_best_description(self, candidates: List[str]) -> Optional[str]:
        best: Optional[str] = None
        best_score = -1e9
        for raw in candidates:
            s = self._clean_inline_text(raw)
            if not s:
                continue
            if len(s) < 40:
                continue
            if len(s) > 1200:
                continue
            score = 0.0
            if _looks_like_sentence(s):
                score += 1.0
            if 80 <= len(s) <= 240:
                score += 1.0
            elif len(s) > 400:
                score -= 0.5
            if score > best_score:
                best_score = score
                best = s
        if not best:
            return None
        return _truncate(best, 240) or None

    def _is_valid_title(self, title: str) -> bool:
        t = (title or "").strip()
        if len(t) < 8:
            return False
        lower = t.lower()
        if any(k in lower for k in self._TITLE_NOISE_HINTS):
            return False
        if self._looks_like_caption_ui_line(t):
            return False
        return True

    def _clean_inline_text(self, text: str) -> str:
        s = _strip_control_chars(text or "")
        s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
        s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
        s = re.sub(r"https?://\S+", "", s)
        s = re.sub(r"\bwww\.[^\s]+", "", s)
        s = _norm_space(s)
        return s

    # -----------------------------
    # Internals: Markdown cleaning (strict body-only)
    # -----------------------------

    def _filter_lines_strict(self, markdown: str) -> List[str]:
        """
        Strict line filter:
        - Drop ALL headings
        - Drop fenced code blocks entirely
        - Drop caption/player UI lines
        - Drop obvious nav/subscription/promotional lines
        - Drop "See all / View all / Show more"
        - Drop list-item link clusters that look like nav blocks
        - Keep only candidate body lines + blank lines (as paragraph boundaries)
        """
        out: List[str] = []

        for raw in markdown.splitlines():
            stripped = raw.strip()

            # preserve blank lines (for paragraph boundaries)
            if not stripped:
                out.append("")
                continue

            # drop headings entirely
            if _is_heading(stripped):
                continue

            s = _norm_space(_strip_control_chars(stripped))
            if not s:
                continue

            lower = s.lower()

            # caption/player UI
            if self._looks_like_caption_ui_line(s):
                continue
            if any(tok in lower for tok in self._UI_TOKENS):
                # keep if it is clearly a sentence-like paragraph (rare)
                if not (_looks_like_sentence(s) and len(s) >= 100):
                    continue

            # explicit see-all controls
            if self._SEE_ALL_PAT.match(s):
                continue

            # markdown table rows/separators should be preserved
            if self._is_possible_markdown_table_line(s):
                out.append(s)
                continue

            # inline noise hints
            if any(h in lower for h in self._INLINE_NOISE_HINTS):
                if not (_looks_like_sentence(s) and len(s) >= 120):
                    continue

            # nav/list-like lines (bullets, numbered lists, short link titles)
            if self._is_nav_listish_line(s):
                continue

            # very short non-sentence lines are almost always junk at this stage
            if self._is_short_noise_line_strict(s):
                continue

            out.append(s)

        # collapse multiple blanks
        compact: List[str] = []
        prev_blank = False
        for ln in out:
            blank = (ln.strip() == "")
            if blank and prev_blank:
                continue
            compact.append(ln)
            prev_blank = blank

        return compact

    def _is_possible_markdown_table_line(self, s: str) -> bool:
        t = s.strip()
        if not t:
            return False
        if _TABLE_SEPARATOR_RE.match(t):
            return True
        if "|" not in t:
            return False
        if t.count("|") < 2:
            return False

        cells = [cell.strip() for cell in t.strip("|").split("|")]
        filled_cells = [cell for cell in cells if cell]
        return len(filled_cells) >= 2

    def _is_markdown_table_line(self, lines: List[str], idx: int) -> bool:
        if idx < 0 or idx >= len(lines):
            return False

        current = lines[idx].strip()
        if not self._is_possible_markdown_table_line(current):
            return False
        if _TABLE_SEPARATOR_RE.match(current):
            return True

        prev_line = lines[idx - 1].strip() if idx > 0 else ""
        next_line = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
        prev_prev_line = lines[idx - 2].strip() if idx > 1 else ""
        next_next_line = lines[idx + 2].strip() if idx + 2 < len(lines) else ""

        if _TABLE_SEPARATOR_RE.match(prev_line) or _TABLE_SEPARATOR_RE.match(next_line):
            return True

        if self._is_possible_markdown_table_line(prev_line):
            if _TABLE_SEPARATOR_RE.match(prev_prev_line) or self._is_possible_markdown_table_line(next_line):
                return True
        if self._is_possible_markdown_table_line(next_line):
            if _TABLE_SEPARATOR_RE.match(next_next_line) or self._is_possible_markdown_table_line(prev_line):
                return True

        return False

    def _is_nav_listish_line(self, s: str) -> bool:
        """
        Detect typical non-body lines:
        - bullet/numbered items (often related links)
        - pipe-separated clusters
        - short title-like lines without punctuation
        """
        t = s.strip()
        if not t:
            return False

        if self._is_possible_markdown_table_line(t):
            return False

        if re.match(r"^[-*•]\s+\S+", t):
            return True
        if re.match(r"^\d+\.\s+\S+", t):
            return True

        if "|" in t and not _looks_like_sentence(t) and len(t) < 260:
            return True

        # "#### Health" 已在 heading 阶段丢弃；这里处理“Health”这种短频道残留
        if len(t) <= 30 and not _looks_like_sentence(t):
            # 单/双 token 的短名词非常像频道
            tokens = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", t)
            if 1 <= len(tokens) <= 3:
                return True

        return False

    def _is_short_noise_line_strict(self, s: str) -> bool:
        """
        Strict short-line removal (more aggressive than original):
        - If not sentence-like and < 80 => drop
        - If sentence-like but < 60 => drop (usually caption/byline)
        """
        t = s.strip()
        # print(len(t), t)
        if not t:
            return True
        if self._is_possible_markdown_table_line(t):
            return False
        if _looks_like_sentence(t):
            return len(t) < 60
        return len(t) < 80

    def _is_body_paragraph(self, p: str) -> bool:
        """
        Decide whether a line is likely main body content.
        High precision:
        - must be sentence-like (punctuation) OR very long
        - must be long enough
        - must have enough word-like tokens
        - must not look like UI/nav leftovers
        """
        s = _norm_space(_strip_control_chars(p or ""))
        if not s:
            return False

        lower = s.lower()

        # UI/menu remnants
        if self._looks_like_caption_ui_line(s):
            return False
        if any(h in lower for h in self._INLINE_NOISE_HINTS):
            return False

        # token/length thresholds
        if len(s) < self._MIN_PARAGRAPH_CHARS:
            return False

        tokens = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", s)
        if len(tokens) < self._MIN_PARAGRAPH_WORDLIKE_TOKENS:
            return False

        # sentence-ness: require punctuation, unless paragraph is very long
        if not _looks_like_sentence(s) and len(s) < max(260, self._MIN_PARAGRAPH_CHARS * 2):
            return False

        # Avoid paragraphs that are basically lists or link clusters
        if self._is_nav_listish_paragraph(s):
            return False

        return True

    def _is_nav_listish_paragraph(self, s: str) -> bool:
        """
        Paragraph-level nav detection:
        - many short fragments separated by '*' or '\n' already merged as ' * '
        - many numbered items
        - low punctuation but many title-like chunks
        """
        t = s.strip()
        if not t:
            return True

        # Many bullet markers that survived merging (rare but happens)
        bullet_like = len(re.findall(r"\s[*•]\s", t))
        if bullet_like >= 2 and not _looks_like_sentence(t):
            return True

        # Many numbered items in one paragraph
        num_like = len(re.findall(r"\b\d+\.\s", t))
        if num_like >= 2:
            return True

        # Very low punctuation density but lots of tokens can indicate nav clusters
        punct_cnt = len(re.findall(r"[.!?。！？]", t))
        if punct_cnt == 0 and len(t) < 600:
            # if it is long but has no sentence punctuation, it's suspicious
            return True

        return False

    def _looks_like_caption_ui_line(self, line: str) -> bool:
        s = line.strip()
        if not s:
            return False

        if self._UI_GLUE_PAT.search(s):
            return True

        lower = s.lower()
        hits = sum(1 for t in self._UI_TOKENS if t in lower)

        has_sentence_punct = bool(re.search(r"[.!?。！？]", s))
        longish = len(s) >= 90

        has_many_percents = len(re.findall(r"\b\d{1,3}%\b", s)) >= 2
        color_hits = sum(1 for c in ("white", "black", "red", "green", "blue", "yellow", "magenta", "cyan") if c in lower)

        if hits >= 4 and longish and not has_sentence_punct:
            return True
        if hits >= 3 and color_hits >= 3 and not has_sentence_punct and longish:
            return True
        if has_many_percents and ("font" in lower or "opacity" in lower) and not has_sentence_punct:
            return True

        return False


def _clean_meta_text(s: str) -> str:
    s = _strip_control_chars(s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s
