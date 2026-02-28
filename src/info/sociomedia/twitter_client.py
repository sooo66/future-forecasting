# -*- coding: utf-8 -*-
"""Twitter/X GraphQL API client (public tweets only)."""
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import quote

from .twitter_types import TweetData

logger = logging.getLogger(__name__)

BEARER_TOKEN = (
    "AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D"
    "1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA"
)

QUERY_IDS = {
    "TweetResultByRestId": "Xl5pC_lBk_gcO2ItU39DQw",
    "TweetDetail": "97JF30KziU00483E_8elBA",
}

FEATURES = {
    "creator_subscriptions_tweet_preview_api_enabled": True,
    "communities_web_enable_tweet_community_results_fetch": True,
    "c9s_tweet_anatomy_moderator_badge_enabled": True,
    "articles_preview_enabled": True,
    "responsive_web_edit_tweet_api_enabled": True,
    "graphql_is_translatable_rweb_tweet_is_translatable_enabled": True,
    "view_counts_everywhere_api_enabled": True,
    "longform_notetweets_consumption_enabled": True,
    "responsive_web_twitter_article_tweet_consumption_enabled": True,
    "tweet_awards_web_tipping_enabled": False,
    "creator_subscriptions_quote_tweet_preview_enabled": False,
    "freedom_of_speech_not_reach_fetch_enabled": True,
    "standardized_nudges_misinfo": True,
    "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": True,
    "rweb_video_timestamps_enabled": True,
    "longform_notetweets_rich_text_read_enabled": True,
    "longform_notetweets_inline_media_enabled": True,
    "rweb_tipjar_consumption_enabled": True,
    "responsive_web_graphql_exclude_directive_enabled": True,
    "verified_phone_label_enabled": False,
    "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
    "responsive_web_graphql_timeline_navigation_enabled": True,
    "responsive_web_enhance_cards_enabled": False,
    "responsive_web_jetfuel_frame": False,
    "responsive_web_grok_analysis_button_from_backend": False,
    "responsive_web_profile_redirect_enabled": False,
    "rweb_video_screen_enabled": False,
    "responsive_web_grok_show_grok_translated_post": False,
    "responsive_web_grok_analyze_post_followups_enabled": False,
    "responsive_web_grok_imagine_annotation_enabled": False,
    "responsive_web_grok_share_attachment_enabled": False,
    "responsive_web_grok_image_annotation_enabled": False,
    "premium_content_api_read_enabled": False,
    "responsive_web_grok_analyze_button_fetch_trends_enabled": False,
    "responsive_web_grok_community_note_auto_translation_is_enabled": False,
    "profile_label_improvements_pcf_label_in_post_enabled": False,
    "tweetypie_unmention_optimization_enabled": True,
}


@dataclass
class TwitterCredentials:
    auth_token: str
    ct0: str
    full_cookie: str = ""


class TwitterClient:
    BASE_URL = "https://x.com/i/api/graphql"
    GUEST_TOKEN_URL = "https://api.x.com/1.1/guest/activate.json"

    def __init__(self) -> None:
        self.guest_token: Optional[str] = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_guest_token(self) -> str:
        if self.guest_token:
            return self.guest_token

        curl_requests = _get_http_client()

        response = _request(
            curl_requests,
            method="post",
            url=self.GUEST_TOKEN_URL,
            headers={"authorization": f"Bearer {BEARER_TOKEN}"},
            timeout=15,
        )

        if response.status_code != 200:
            raise RuntimeError(f"guest token failed: HTTP {response.status_code}")

        data = response.json()
        self.guest_token = data.get("guest_token", "")
        if not self.guest_token:
            raise RuntimeError("guest token missing")

        self.logger.debug("Got guest token")
        return self.guest_token

    def _build_guest_headers(self) -> Dict[str, str]:
        return {
            "authorization": f"Bearer {BEARER_TOKEN}",
            "x-guest-token": self._get_guest_token(),
            "user-agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "origin": "https://x.com",
            "referer": "https://x.com/",
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
        }

    def _build_tweet_result_url(self, tweet_id: str) -> str:
        variables = {
            "tweetId": tweet_id,
            "withCommunity": False,
            "includePromotedContent": False,
            "withVoice": False,
        }

        params = {
            "variables": json.dumps(variables, separators=(",", ":")),
            "features": json.dumps(FEATURES, separators=(",", ":")),
        }

        query_string = "&".join(f"{k}={quote(v)}" for k, v in params.items())
        query_id = QUERY_IDS["TweetResultByRestId"]
        return f"{self.BASE_URL}/{query_id}/TweetResultByRestId?{query_string}"

    def get_tweet(self, tweet_id: str) -> TweetData:
        curl_requests = _get_http_client()

        url = self._build_tweet_result_url(tweet_id)
        self.logger.info(f"Fetching tweet {tweet_id}")

        headers = self._build_guest_headers()
        response = _request(
            curl_requests,
            method="get",
            url=url,
            headers=headers,
            timeout=15,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Request failed: HTTP {response.status_code}")

        data = response.json()
        tweet_result = data.get("data", {}).get("tweetResult", {})
        if tweet_result and tweet_result.get("result"):
            return TweetData.from_tweet_result(tweet_result)

        if "errors" in data:
            error_msg = data["errors"][0].get("message", "Unknown error")
            raise ValueError(f"API error: {error_msg}")

        raise ValueError(f"Tweet not found: {tweet_id}")


def _get_http_client():
    try:
        from curl_cffi import requests as curl_requests
        return curl_requests
    except Exception:
        import requests as curl_requests
        return curl_requests


def _request(client, *, method: str, url: str, headers: dict, timeout: int):
    supports_impersonate = client.__name__.startswith("curl_cffi")
    kwargs = {"headers": headers, "timeout": timeout}
    if supports_impersonate:
        kwargs["impersonate"] = "chrome"
    if method == "get":
        return client.get(url, **kwargs)
    if method == "post":
        return client.post(url, **kwargs)
    raise ValueError(f"Unsupported method: {method}")


def extract_tweet_id(url: str) -> str:
    patterns = [
        r"(?:twitter|x)\.com/\w+/status/(\d+)",
        r"(?:twitter|x)\.com/i/web/status/(\d+)",
        r"/status/(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError(f"Cannot extract tweet ID: {url}")
