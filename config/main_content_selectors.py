"""Domain-specific main content selectors.

Defaults are intentionally generic and stable to avoid brittle selectors.
Each domain can extend or override via DOMAIN_SELECTORS.
"""

DEFAULT_TARGETS = [
    "article",
    "main article",
    "[itemprop='articleBody']",
    "[data-article-body]",
    "[class*='article__content']",
    "[class*='article-body']",
    "[class*='article-content']",
    "[class*='content__body']",
    "[class*='story-body']",
    "[class*='story__body']",
    "[id*='article-body']",
    "[id*='story-body']",
]

DEFAULT_EXCLUDE = [
    "video",
    "iframe",
    "picture",
    "figure",
    "img",
    "svg",
    "source",
    "canvas",
    "audio",
    "aside",
    "[class*='video']",
    "[id*='video']",
    "[class*='player']",
    "[id*='player']",
    "[class*='ad-']",
    "[class^='ad']",
    "[class*='ads']",
    "[id*='ad-']",
    "[id^='ad']",
    "[id*='ads']",
    "[class*='advert']",
    "[id*='advert']",
    "[class*='sponsor']",
    "[id*='sponsor']",
    "[class*='promo']",
    "[id*='promo']",
    "[class*='sidebar']",
    "[id*='sidebar']",
    "[class*='social']",
    "[id*='social']",
    "[class*='share']",
    "[id*='share']",
    "[class*='related']",
    "[id*='related']",
    "[class*='recommend']",
    "[id*='recommend']",
    "[class*='newsletter']",
    "[id*='newsletter']",
    "[class*='subscribe']",
    "[id*='subscribe']",
    "[class*='comment']",
    "[id*='comment']",
    "[class*='cookie']",
    "[id*='cookie']",
    "[class*='popup']",
    "[id*='popup']",
    "[class*='overlay']",
    "[id*='overlay']",
    "[class*='modal']",
    "[id*='modal']",
    "aside",
    "nav",
    "footer",
]

DOMAIN_SELECTORS = {
    "justthenews.com": {
        "targets": ["[class*='node__text']"],
        "exclude": [".video--content"],
    },
    "zerohedge.com": {},
    "nypost.com": {
        "targets": ["[class*='entry-content']"],
        "exclude": [".featured-image", "figcaption"],
    },
    "bbc.com": {},
    "reason.com": {},
    "foxbusiness.com": {
        "exclude": [".featured-video", ".video-ct", "[class*='media']"],
    },
    "cnbc.com": {
        "exclude": [".InlineImage-wrapper", "[id^='ArticleBody-InlineImage-']"],
    },
    "us.cnn.com": {
        "targets": ["main"],
        "exclude": ["label", "div.image__lede.article__lede-wrapper", ".article__lede-wrapper"],
    },
    "edition.cnn.com": {
        "exclude": ["div.image__lede.article__lede-wrapper", ".article__lede-wrapper"],
    },
    "newsweek.com": {
        "exclude": ["#dfp-ad-inarticle1-wrapper", "[id^='dfp-ad-']"],
    },
    "cbsnews.com": {
        "exclude": ["#mpu-plus-outstream-middle", "[id*='outstream']"],
    },
    "csmonitor.com": {
        "targets": ["[class*='eza-body']"],
        "exclude": ["[class*='offcanvas']", "[class*='megawhy-block']", "[class*='story-foot']"],
    },
    "time.com": {},
    "dailymail.com": {
        "targets": ["[id*='article-text']"],
        "exclude": ["[class*='ob-smartfeed-joker']", "[class^='mol-style']", "span"],
    },
    "propublica.org": {
        "exclude": ["[class*='caption']","[class*='credit']", "div.wp-block-propublica-reporting-highlights", "[class*='reporting-highlights']"],
    },
}
