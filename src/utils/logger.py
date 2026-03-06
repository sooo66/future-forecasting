"""日志配置模块"""
import sys
from pathlib import Path

from loguru import logger

from .config import Config

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional at import time
    tqdm = None  # type: ignore[assignment]


def _console_sink(message: object) -> None:
    text = str(message).rstrip("\n")
    if not text:
        return
    if tqdm is not None:
        tqdm.write(text, file=sys.stdout)
        return
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def setup_logger(config: Config, *, verbose: bool = False):
    """配置日志系统"""
    log_dir = config.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 移除默认处理器
    logger.remove()

    # 调整控制台级别配色，避免 INFO 与进度条默认白色混淆
    logger.level("DEBUG", color="<blue>")
    logger.level("INFO", color="<cyan>")
    logger.level("SUCCESS", color="<green>")
    logger.level("WARNING", color="<yellow>")
    logger.level("ERROR", color="<red>")
    logger.level("CRITICAL", color="<red><bold>")
    
    # 添加控制台输出（默认 INFO，verbose 时 DEBUG）
    logger.add(
        _console_sink,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="DEBUG" if verbose else "INFO",
        colorize=True,
        filter=lambda record: record["extra"].get("trace") is not True,
    )
    
    # 添加文件输出（所有级别）
    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="00:00",
        retention="30 days",
        compression="zip"
    )

    # 精简爬虫追踪日志（JSONL）：仅记录显式标记 trace=True 的事件，便于定位失败链路/LLM I/O
    logger.add(
        log_dir / "crawl_trace_{time:YYYY-MM-DD}.jsonl",
        level="DEBUG",
        rotation="00:00",
        retention="30 days",
        compression="zip",
        serialize=True,
        filter=lambda record: record["extra"].get("trace") is True,
    )
    
    # 添加错误日志文件
    logger.add(
        log_dir / "error_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="00:00",
        retention="90 days",
        compression="zip"
    )
    
    return logger
