"""GDELT GKG 数据解析模块"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from loguru import logger
from tqdm import tqdm

from utils.config import Config


class GDELTParser:
    """GDELT GKG 数据解析器
    
    根据 GDELT GKG 2.0 格式规范，GKG CSV 文件是制表符分隔的，包含以下字段：
    - GKGRECORDID: 记录ID
    - DATE: 日期时间（YYYYMMDDHHMMSS）
    - SourceCollectionIdentifier: 来源集合标识
    - SourceCommonName: 来源通用名称
    - DocumentIdentifier: 文档标识
    - Counts: 计数信息
    - V2Counts: V2计数信息
    - Themes: 主题（分号分隔）
    - V2Themes: V2主题
    - Locations: 位置信息
    - V2Locations: V2位置信息
    - Persons: 人物
    - V2Persons: V2人物
    - Organizations: 组织
    - V2Organizations: V2组织
    - V2Tone: V2语调
    - Dates: 日期
    - V2Dates: V2日期
    - GCAM: GCAM信息
    - SharingImage: 共享图片
    - RelatedImages: 相关图片
    - SocialImageEmbeds: 社交图片嵌入
    - SocialVideoEmbeds: 社交视频嵌入
    - Quotations: 引用
    - V2Quotations: V2引用
    - AllNames: 所有名称
    - Amounts: 金额
    - V2Amounts: V2金额
    - TranslationInfo: 翻译信息
    - Extras: 额外信息
    - SOURCES: 来源（分号分隔）
    - SOURCEURLS: 来源URL（分号分隔）
    """
    
    # GKG CSV 字段索引（根据实际文件格式）
    # 注意：GKG 文件是制表符分隔的，实际格式为 11 列
    # 实际字段：DATE, NUMARTS, COUNTS, THEMES, LOCATIONS, PERSONS, ORGANIZATIONS, TONE, CAMEOEVENTIDS, SOURCES, SOURCEURLS
    FIELD_NAMES = [
        "DATE", "NUMARTS", "COUNTS", "THEMES", "LOCATIONS", 
        "PERSONS", "ORGANIZATIONS", "TONE", "CAMEOEVENTIDS", 
        "SOURCES", "SOURCEURLS"
    ]
    
    def __init__(self, config: Config):
        self.config = config
        self.raw_data_dir = config.raw_data_dir
        self.processed_data_dir = config.processed_data_dir
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def _parse_gkg_file(self, csv_path: Path) -> Optional[pd.DataFrame]:
        """解析单个 GKG CSV 文件
        
        只提取需要的字段：
        - DATE: 日期（YYYYMMDD 格式）
        - THEMES: 主题（用于后续生成 tags）
        - SOURCES: 来源
        - SOURCEURLS: 来源URL
        
        注意：实际 GKG 文件格式为 11 列，没有 GKGRECORDID 字段
        """
        try:
            # 使用 pandas 读取，指定分隔符为制表符
            # 由于文件可能很大，使用分块读取
            chunks = []
            chunk_size = 10000
            
            logger.debug(f"开始解析 {csv_path.name}")
            
            for chunk in pd.read_csv(
                csv_path,
                sep='\t',
                names=self.FIELD_NAMES,
                usecols=["DATE", "THEMES", "SOURCES", "SOURCEURLS"],
                dtype=str,
                na_values=[''],
                keep_default_na=False,
                chunksize=chunk_size,
                low_memory=False
            ):
                # 过滤掉空行（只检查 DATE 字段）
                chunk = chunk.dropna(subset=["DATE"])
                # 过滤掉 DATE 为空字符串的行
                chunk = chunk[chunk["DATE"].str.strip() != ""]
                if len(chunk) > 0:
                    chunks.append(chunk)
            
            if not chunks:
                logger.warning(f"{csv_path.name} 没有有效数据")
                return None
            
            df = pd.concat(chunks, ignore_index=True)
            
            # 解析日期（格式为 YYYYMMDD，不是 YYYYMMDDHHMMSS）
            # 先尝试完整格式，如果失败则尝试日期格式
            df['DATE_PARSED'] = pd.to_datetime(df['DATE'], format='%Y%m%d', errors='coerce')
            # 如果解析失败，尝试完整格式
            if df['DATE_PARSED'].isna().any():
                df['DATE_PARSED'] = pd.to_datetime(df['DATE'], format='%Y%m%d%H%M%S', errors='coerce')
            
            # 提取日期部分（用于后续处理）
            df['gkg_date'] = df['DATE_PARSED'].dt.strftime('%Y-%m-%d')
            # 如果日期解析失败，使用原始 DATE 字段的前 8 位（YYYYMMDD）
            df['gkg_date'] = df['gkg_date'].fillna(df['DATE'].str[:8].apply(
                lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}" if len(x) >= 8 else None
            ))
            
            # 创建记录ID（使用行号 + 日期作为唯一标识）
            # 同时创建两种格式以兼容不同模块的需求
            df['GKGRECORDID'] = df.index.astype(str) + '_' + df['DATE'].astype(str)
            df['gkg_record_id'] = df['GKGRECORDID']  # 小写下划线格式，用于 URL 池构建器
            
            logger.info(f"成功解析 {csv_path.name}，共 {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"解析 {csv_path.name} 时发生错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def parse_all(self) -> Dict[str, int]:
        """解析所有已下载的 GKG 文件"""
        csv_files = list(self.raw_data_dir.glob("*.gkg.csv"))
        
        if not csv_files:
            logger.warning("没有找到 GKG CSV 文件")
            return {"total": 0, "success": 0, "failed": 0}
        
        logger.info(f"开始解析 {len(csv_files)} 个 GKG 文件")
        
        success_count = 0
        fail_count = 0
        total_records = 0
        
        for csv_file in tqdm(csv_files, desc="解析 GKG 文件"):
            df = self._parse_gkg_file(csv_file)
            
            if df is not None and len(df) > 0:
                # 保存为 Parquet 格式（更高效）
                date_str = csv_file.stem.split(".")[0]
                output_path = self.processed_data_dir / f"{date_str}_gkg.parquet"
                df.to_parquet(output_path, index=False, compression='snappy')
                
                # 删除原始 CSV 文件以节省空间
                csv_file.unlink()
                
                success_count += 1
                total_records += len(df)
                logger.debug(f"已保存 {output_path}，包含 {len(df)} 条记录")
            else:
                fail_count += 1
        
        logger.info(f"GKG 解析完成: 成功 {success_count} 个文件，失败 {fail_count} 个文件，共 {total_records} 条记录")
        
        return {
            "total": len(csv_files),
            "success": success_count,
            "failed": fail_count,
            "records": total_records
        }

