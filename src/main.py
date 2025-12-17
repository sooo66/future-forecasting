"""主入口 CLI"""
import asyncio
import argparse
from pathlib import Path
from loguru import logger

from utils.config import Config
from utils.logger import setup_logger
from gdelt_downloader import GDELTDownloader, GDELTParser
from url_pool_builder import URLPoolBuilder
from crawler import NewsCrawler


def download_gkg(args):
    """下载 GKG 数据"""
    config = Config(args.config)
    setup_logger(config)
    
    logger.info("开始下载 GDELT GKG 数据")
    
    downloader = GDELTDownloader(config)
    result = downloader.download()
    
    logger.info(f"下载完成: {result}")
    return result


def parse_gkg(args):
    """解析 GKG 数据"""
    config = Config(args.config)
    setup_logger(config)
    
    logger.info("开始解析 GDELT GKG 数据")
    
    parser = GDELTParser(config)
    result = parser.parse_all()
    
    logger.info(f"解析完成: {result}")
    return result


def build_url_pool(args):
    """构建 URL 池"""
    config = Config(args.config)
    setup_logger(config)
    
    logger.info("开始构建 URL 池")
    
    builder = URLPoolBuilder(config)
    result = builder.build()
    
    # 显示统计信息
    stats = builder.get_statistics()
    logger.info(f"URL 池统计: {stats}")
    
    return result


def crawl(args):
    """爬取新闻"""
    config = Config(args.config)
    setup_logger(config)
    
    logger.info("开始爬取新闻")
    
    builder = URLPoolBuilder(config)
    crawler = NewsCrawler(config, builder)
    
    limit = args.limit if hasattr(args, 'limit') and args.limit else None
    
    # 运行异步爬取
    records = asyncio.run(crawler.crawl(limit=limit))
    records = records or []  # 保底避免 None
    
    logger.info(f"爬取完成，共获得 {len(records)} 条记录")
    return records


def full_pipeline(args):
    """完整流程"""
    config = Config(args.config)
    setup_logger(config)
    
    logger.info("开始完整流程")
    
    # 1. 下载 GKG 数据
    logger.info("=" * 50)
    logger.info("步骤 1/4: 下载 GKG 数据")
    logger.info("=" * 50)
    downloader = GDELTDownloader(config)
    download_result = downloader.download()
    logger.info(f"下载结果: {download_result}")
    
    # 2. 解析 GKG 数据
    logger.info("=" * 50)
    logger.info("步骤 2/4: 解析 GKG 数据")
    logger.info("=" * 50)
    parser = GDELTParser(config)
    parse_result = parser.parse_all()
    logger.info(f"解析结果: {parse_result}")
    
    # 3. 构建 URL 池
    logger.info("=" * 50)
    logger.info("步骤 3/4: 构建 URL 池")
    logger.info("=" * 50)
    builder = URLPoolBuilder(config)
    build_result = builder.build()
    stats = builder.get_statistics()
    logger.info(f"构建结果: {build_result}")
    logger.info(f"URL 池统计: {stats}")
    
    # 4. 爬取新闻
    logger.info("=" * 50)
    logger.info("步骤 4/4: 爬取新闻")
    logger.info("=" * 50)
    crawler = NewsCrawler(config, builder)
    limit = args.limit if hasattr(args, 'limit') and args.limit else None
    records = asyncio.run(crawler.crawl(limit=limit))
    
    logger.info("=" * 50)
    logger.info("完整流程执行完成")
    logger.info("=" * 50)
    logger.info(f"最终统计:")
    logger.info(f"  - 下载文件: {download_result['success']}/{download_result['total']}")
    logger.info(f"  - 解析记录: {parse_result['records']}")
    logger.info(f"  - URL 池: {stats['total']} (待爬取: {stats['pending']})")
    logger.info(f"  - 爬取记录: {len(records)}")
    
    return {
        "download": download_result,
        "parse": parse_result,
        "build": build_result,
        "crawl": len(records)
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="新闻爬虫系统 - 从 GDELT GKG 构建新闻 URL 池并爬取结构化数据"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.toml",
        help="配置文件路径（默认: config/settings.toml）"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # download-gkg 命令
    subparsers.add_parser("download-gkg", help="下载 GDELT GKG 数据")
    
    # parse-gkg 命令
    subparsers.add_parser("parse-gkg", help="解析 GDELT GKG 数据")
    
    # build-url-pool 命令
    subparsers.add_parser("build-url-pool", help="构建 URL 池")
    
    # crawl 命令
    crawl_parser = subparsers.add_parser("crawl", help="爬取新闻")
    crawl_parser.add_argument(
        "--limit",
        type=int,
        help="限制爬取的 URL 数量（用于测试）"
    )
    
    # full-pipeline 命令
    pipeline_parser = subparsers.add_parser("full-pipeline", help="执行完整流程")
    pipeline_parser.add_argument(
        "--limit",
        type=int,
        help="限制爬取的 URL 数量（用于测试）"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "download-gkg":
            download_gkg(args)
        elif args.command == "parse-gkg":
            parse_gkg(args)
        elif args.command == "build-url-pool":
            build_url_pool(args)
        elif args.command == "crawl":
            crawl(args)
        elif args.command == "full-pipeline":
            full_pipeline(args)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        logger.info("用户中断操作")
    except Exception as e:
        logger.exception(f"执行失败: {e}")
        raise


if __name__ == "__main__":
    main()
