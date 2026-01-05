# from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, ProxyConfig, user_agent_generator
# from crawl4ai.content_filter_strategy import PruningContentFilter
# from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
# import time
# import json
# from crawl4ai import CosineStrategy

# md_generator = DefaultMarkdownGenerator(
#     content_filter=PruningContentFilter(threshold=0.4, threshold_type="fixed")
# )

# extractor_strategy = CosineStrategy(
#     semantic_filter="main article content",
#     word_count_threshold=30, 
#     top_k=1,
#     sim_threshold=0.68
# )


# browse_conf = BrowserConfig(
#     headless=False,
#     user_agent_mode="random",
#     text_mode=True,
#     light_mode=True,
#     extra_args=["--disable-extensions"],
#     enable_stealth=True
# )

# config = CrawlerRunConfig(
#     magic=True,
#     only_text=True,
#     extraction_strategy=extractor_strategy,
#     # markdown_generator=md_generator,
#     exclude_external_links=True,
#     exclude_internal_links=True,
#     exclude_social_media_links=True,
#     exclude_all_images=True,
#     user_agent_mode="random",
# )

# urls = ['https://www.zerohedge.com/markets/pepsi-cuts-deal-activist-plans-product-overhaul-and-layoffs', 'https://nypost.com/2025/12/09/us-news/fbi-agents-fired-after-kneeling-during-2020-blm-protest-sue-to-get-their-jobs-back/', 'https://www.bloomberg.com:443/news/articles/2025-12-08/zelenskiy-signals-no-security-breakthrough-in-london-peace-talks', 'https://www.bbc.com/news/articles/cwyp98vxk78o', 'https://www.washingtonexaminer.com/news/senate/3911442/john-thune-obamacare-subsidies-democratic-bill/', 'https://www.forbes.com/councils/forbesbusinesscouncil/2025/12/09/emerging-markets-as-a-frontier-for-global-payments-technology/', 'https://reason.com/2025/12/08/donald-trump-says-hell-be-involved-in-choosing-who-gets-to-merge-with-warner-bros/', 'https://www.foxbusiness.com/lifestyle/powerball-jackpot-surges-930m', 'https://www.cnbc.com/2025/12/09/trump-trade-rep-changes-china-soybean-purchase-timeline-cites-discrepancy.html', 'https://us.cnn.com/2025/12/09/food/mccormick-flavor-of-the-year-2026', 'https://www.newsweek.com/absolutely-american-story-of-how-aretha-franklins-natural-woman-happened-11174707', 'https://www.theepochtimes.com/business/homeowners-face-average-73-percent-higher-costs-if-they-relocate-report-5955819', 'https://www.cbsnews.com/news/dnc-democratic-chair-ken-martin-2025-wins-midterm-cycle-interview/', 'https://www.nbcnews.com/tech/social-media/28-us-teens-say-use-ai-chatbots-daily-poll-says-rcna248133', 'https://www.nytimes.com/interactive/2025/12/09/us/crypto-casinos-gambling-streamers.html', 'https://www.csmonitor.com/World/Africa/2025/1209/Somalis-Minnesota-garbage-Trump', 'https://www.dailymail.co.uk/buyline/article-15347351/Item-sells-minute-globally-transforms-eye-bags-SECONDS.html', 'https://justthenews.com/politics-policy/trump-zelensky-hasnt-read-ukraine-peace-proposal', 'https://time.com/7339363/us-civil-liberties-authoritarian-shift-civicus-trump/', 'https://www.propublica.org/article/federally-qualified-health-centers-unpaid-bills-lawsuits']

# results = []

# async def main():
#     async with AsyncWebCrawler(config=browse_conf) as crawler:
#         total_st_time = time.time()
#         for idx, url in enumerate(urls):
#             st_time = time.time()
#             result = await crawler.arun(url, config=config)
#             end_time = time.time()
#             print(f"=== Result for URL index {idx} ===")
#             print()
#             # print("Raw Markdown length:", len(result.markdown.raw_markdown))
#             # print("Fit Markdown length:", len(result.markdown.fit_markdown))
#             r = {
#                 "idx": idx,
#                 "url": url,
#                 # "fit_markdown": result.markdown.fit_markdown,
#                 "extraction_content": result.extracted_content,
#             }
#             with open("crawl4ai_results_cos.jsonl", "a", encoding="utf-8") as f:
#                 f.write(json.dumps(r, ensure_ascii=False, indent=2) + "\n")
#             results.append(r)
#         total_end_time = time.time()
#         total_time = total_end_time - total_st_time
#         print(f"Average time per URL: {total_time / len(urls):.2f} seconds")
#         with open("crawl4ai_results.json", "w", encoding="utf-8") as f:
#             f.write(json.dumps(results, ensure_ascii=False, indent=2))

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())

import json
import re
from dataclasses import dataclass
from typing import Optional, List
from urllib.parse import urlparse

from bs4 import BeautifulSoup


_CAPTION_UI_PATTERNS = (
    r"(?:caption|subtitles?)\s*(?:area\s*)?background\s*color[\s\S]{0,500}?font\s*family[\s\S]{0,200}?",
    r"text\s*background\s*color[\s\S]{0,500}?font\s*family[\s\S]{0,200}?",
    r"text\s*edge\s*style[\s\S]{0,500}?font\s*family[\s\S]{0,200}?",
    r"font\s*size\s*\d+%[\s\S]{0,500}?font\s*family[\s\S]{0,200}?",
)


def _remove_caption_ui_text(text: str) -> str:
    if not text:
        return ""

    for pat in _CAPTION_UI_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.I)

    segments = re.split(r"(?<=[.!?])\s+|\s+\|\s+|\s{2,}", text)
    kept: List[str] = []
    for seg in segments:
        s = seg.strip()
        if not s:
            continue
        lower = s.lower()
        hits = sum(1 for kw in _CAPTION_UI_PATTERNS if kw in lower)
        if hits >= 3 and len(s) <= 600:
            continue
        kept.append(s)

    return " ".join(kept)


import re
from typing import List

# 这些词对“字幕/播放器设置 UI”具有高区分度
_UI_TOKENS = (
    "text color", "background color", "caption", "caption area",
    "opacity", "font size", "font family", "text edge", "edge style",
    "raised", "depressed", "uniform", "drop shadow",
    "proportional", "sans-serif", "monospace", "serif",
    "white", "black", "red", "green", "blue", "yellow", "magenta", "cyan",
    "transparent", "opaque", "semi-transparent",
)

# 一些“粘连”形态的关键片段（专治 ColorWhite / OpacityOpaque 这种）
_UI_GLUE_PAT = re.compile(
    r"(Text\s*Color|Opacity|Font\s*Size|Font\s*Family|Text\s*Background\s*Color|Caption\s*Area\s*Background\s*Color)"
    r"(White|Black|Red|Green|Blue|Yellow|Magenta|Cyan|Opaque|Transparent|Semi-Transparent|\d{1,3}%|\w+)",
    flags=re.I
)

def remove_caption_ui_lines(markdown_or_text: str) -> str:
    """
    Remove caption/player UI garbage lines that often appear as a single long line.
    Safe to run on markdown/text that has already been extracted.
    """
    if not markdown_or_text:
        return ""

    kept: List[str] = []
    for raw in markdown_or_text.splitlines():
        line = raw.strip()
        if not line:
            continue

        lower = line.lower()

        # Strong signal 1: "glued" UI tokens (ColorWhite, OpacityOpaque, etc.)
        if _UI_GLUE_PAT.search(line):
            continue

        # Strong signal 2: line contains many UI tokens and is "UI-like" (dense, not sentence-like)
        hits = sum(1 for t in _UI_TOKENS if t in lower)

        # Heuristic: UI lines tend to have many short tokens, few sentence markers.
        has_sentence_punct = any(p in line for p in ".!?")
        is_very_long = len(line) >= 120

        if hits >= 6 and is_very_long and not has_sentence_punct:
            continue

        kept.append(raw.rstrip())

    return "\n".join(kept)

_UI_TOKENS = (
    "text color", "background color", "caption", "caption area",
    "opacity", "font size", "font family", "text edge", "edge style",
    "raised", "depressed", "uniform", "drop shadow",
    "proportional", "sans-serif", "monospace", "serif",
    "white", "black", "red", "green", "blue", "yellow", "magenta", "cyan",
    "transparent", "opaque", "semi-transparent",
)

# 一些“粘连”形态的关键片段（专治 ColorWhite / OpacityOpaque 这种）
_UI_GLUE_PAT = re.compile(
    r"(Text\s*Color|Opacity|Font\s*Size|Font\s*Family|Text\s*Background\s*Color|Caption\s*Area\s*Background\s*Color)"
    r"(White|Black|Red|Green|Blue|Yellow|Magenta|Cyan|Opaque|Transparent|Semi-Transparent|\d{1,3}%|\w+)",
    flags=re.I
)


def _is_caption_ui_line(line: str) -> bool:
    lower = line.lower()
    if _UI_GLUE_PAT.search(line):
        return True
    hits = sum(1 for t in _UI_TOKENS if t in lower)
    has_sentence_punct = any(p in line for p in ".!?")
    is_very_long = len(line) >= 120
    if hits >= 6 and is_very_long and not has_sentence_punct:
        return True
    return False


text = '''
#  Trump: Zelensky hasn't read Ukraine peace proposal\nText ColorWhite Black Red Green Blue Yellow Magenta CyanOpacityOpaque Semi-Transparent Text Background ColorBlack White Red Green Blue Yellow Magenta CyanOpacityOpaque Semi-Transparent Transparent Caption Area Background ColorBlack White Red Green Blue Yellow Magenta CyanOpacityTransparent Semi-Transparent Opaque\nFont Size 50% 75% 100% 125% 150% 175% 200% 300% 400% Text Edge Style None Raised Depressed Uniform Drop shadow Font Family Proportional Sans-Serif Monospace Sans-Serif Proportional Serif Monospace Serif Casual Script Small Caps\nPresident Donald Trump this week told Politico that Ukrainian President Volodymyr Zelensky had not read his proposed peace plan to end the war with Russia.\nTrump campaigned on ending the Ukraine War within 24 hours of taking office, a deadline that has long passed. His plan calls for Ukraine ceding territory to the Russians, which Zelensky refuses to do.\nAt present, battles rage for the frontline cities of Kupiansk, Lyman, and Siversk. Meanwhile, Russian forces have largely captured Pokrovsk and encircled the garrison of Myrnograd.
'''

for line in text.splitlines():
    if not _is_caption_ui_line(line):
        print(line)
# print(result)
