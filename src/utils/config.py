"""配置管理模块"""
import sys
from pathlib import Path
from typing import Dict, Any, List
from loguru import logger

# 兼容性处理：Python 3.11+ 使用内置 tomllib，否则使用 tomli
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError("Python < 3.11 需要安装 tomli: pip install tomli")


class Config:
    """配置管理器"""
    
    def __init__(self, config_path: "str | Path | dict" = "config/settings.toml"):
        """初始化配置
        
        - 默认从 `config/settings.toml` 读取配置文件；
        - 在测试或脚本中可以直接传入一个 dict，避免依赖磁盘上的 TOML 文件。
        """
        # 直接传入 dict 时，跳过文件读取，便于单元测试
        if isinstance(config_path, dict):
            self.config_path = None
            self._config = config_path
            return
        
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(self.config_path, "rb") as f:
            self._config = tomllib.load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的嵌套键"""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value
    
    @property
    def start_date(self) -> str:
        return self.get("general.start_date")
    
    @property
    def end_date(self) -> str:
        return self.get("general.end_date")
    
    @property
    def concurrent_downloads(self) -> int:
        return self.get("general.concurrent_downloads", 3)
    
    @property
    def concurrent_crawls(self) -> int:
        return self.get("general.concurrent_crawls", 5)
    
    @property
    def per_domain_concurrency(self) -> int:
        return self.get("general.per_domain_concurrency", 2)
    
    @property
    def retry_attempts(self) -> int:
        return self.get("general.retry_attempts", 3)
    
    @property
    def retry_delay_base(self) -> float:
        return self.get("general.retry_delay_base", 2.0)
    
    @property
    def request_delay_min(self) -> float:
        return self.get("general.request_delay_min", 1.0)
    
    @property
    def request_delay_max(self) -> float:
        return self.get("general.request_delay_max", 3.0)
    
    @property
    def test_mode(self) -> bool:
        return self.get("general.test_mode", False)
    
    @property
    def test_url_limit(self) -> int:
        return self.get("general.test_url_limit", 100)
    
    @property
    def user_agents(self) -> List[str]:
        return self.get("general.user_agents", [])
    
    @property
    def proxy_sources(self) -> List[str]:
        return self.get("general.proxy_sources", [])
    
    @property
    def use_proxy(self) -> bool:
        return self.get("general.use_proxy", True)
    
    @property
    def use_browser(self) -> bool:
        return self.get("general.use_browser", False)
    
    @property
    def raw_data_dir(self) -> Path:
        return Path(self.get("paths.raw_data_dir", "data/raw/gkg"))
    
    @property
    def processed_data_dir(self) -> Path:
        return Path(self.get("paths.processed_data_dir", "data/processed/records"))
    
    @property
    def url_pool_db(self) -> Path:
        return Path(self.get("paths.url_pool_db", "data/url_pool/url_pool.db"))
    
    @property
    def log_dir(self) -> Path:
        return Path(self.get("paths.log_dir", "logs"))
    
    @property
    def media_domains(self) -> Dict[str, str]:
        """返回域名到媒体名称的映射"""
        domains = {}
        for item in self.get("whitelist.media_domains", []):
            domains[item["domain"]] = item["name"]
        return domains
    
    @property
    def whitelist_domains(self) -> set:
        """返回域名集合"""
        return set(self.media_domains.keys())
    
    @property
    def browser_headless(self) -> bool:
        return self.get("browser.headless", True)
    
    @property
    def browser_timeout(self) -> int:
        return self.get("browser.timeout", 30)

    # LLM 提取相关配置
    @property
    def use_llm_extraction(self) -> bool:
        return self.get("llm.use_llm_extraction", False)

    @property
    def llm_mode(self) -> str:
        # 可选: always / fallback
        return self.get("llm.mode", "fallback")

    @property
    def llm_provider(self) -> str:
        return self.get("llm.provider", "openai/gpt-4o-mini")

    @property
    def llm_api_key_env(self) -> str:
        return self.get("llm.api_key_env", "OPENAI_API_KEY")

    @property
    def llm_instruction(self) -> str:
        return self.get(
            "llm.instruction",
            (
                "Return a single JSON object with fields: title (optional), summary (optional), content (required), "
                "published_at (optional), language (optional). "
                "Copy the main article body VERBATIM into content — no summarize/paraphrasing/translation/rewriting. "
                "If HTML has no description/summary, generate a concise summary; otherwise preserve provided summary. "
                "Preserve original wording/order in content, omit navigation/ads. "
                "If any field is unavailable, set it to null; do not fabricate."
            ),
        )

    @property
    def llm_chunk_token_threshold(self) -> int:
        return self.get("llm.chunk_token_threshold", 1200)

    @property
    def llm_overlap_rate(self) -> float:
        return self.get("llm.overlap_rate", 0.1)

    @property
    def llm_apply_chunking(self) -> bool:
        return self.get("llm.apply_chunking", True)

    @property
    def llm_input_format(self) -> str:
        return self.get("llm.input_format", "fit_markdown")

    @property
    def llm_temperature(self) -> float:
        return self.get("llm.temperature", 0.0)

    @property
    def llm_max_output_tokens(self) -> int:
        return self.get("llm.max_output_tokens", 1200)
