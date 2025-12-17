"""GDELT GKG 数据下载模块"""
import os
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from loguru import logger
from tqdm import tqdm

from utils.config import Config


class GDELTDownloader:
    """GDELT GKG 数据下载器"""
    
    BASE_URL = "http://data.gdeltproject.org/gkg"
    
    def __init__(self, config: Config):
        self.config = config
        self.raw_data_dir = config.raw_data_dir
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.concurrent_downloads = config.concurrent_downloads
        
        # 记录已下载的文件（用于断点续传）
        self.downloaded_files: Set[str] = self._load_downloaded_files()
    
    def _load_downloaded_files(self) -> Set[str]:
        """加载已下载的文件列表"""
        downloaded = set()
        for file in self.raw_data_dir.glob("*.gkg.csv"):
            # 提取日期部分（YYYYMMDD）
            date_str = file.stem.split(".")[0]
            downloaded.add(date_str)
        return downloaded
    
    def _get_date_range(self) -> List[datetime]:
        """获取日期范围"""
        start = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.config.end_date, "%Y-%m-%d")
        
        dates = []
        current = start
        while current <= end:
            dates.append(current)
            current += timedelta(days=1)
        
        return dates
    
    def _download_file(self, date: datetime) -> bool:
        """下载单个日期的 GKG 文件"""
        date_str = date.strftime("%Y%m%d")
        
        # 检查是否已下载
        if date_str in self.downloaded_files:
            logger.debug(f"{date_str} 的 GKG 文件已存在，跳过下载")
            return True
        
        url = f"{self.BASE_URL}/{date_str}.gkg.csv.zip"
        zip_path = self.raw_data_dir / f"{date_str}.gkg.csv.zip"
        csv_path = self.raw_data_dir / f"{date_str}.gkg.csv"
        
        try:
            # 下载文件
            response = requests.get(url, stream=True, timeout=60)
            
            if response.status_code == 404:
                logger.warning(f"{date_str} 的 GKG 文件不存在（404）")
                return False
            
            if response.status_code != 200:
                logger.warning(f"{date_str} 的 GKG 文件下载失败，状态码: {response.status_code}")
                return False
            
            # 保存 ZIP 文件
            total_size = int(response.headers.get('content-length', 0))
            with open(zip_path, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
            
            # 解压文件
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # GKG 文件解压后通常只有一个 CSV 文件
                    zip_ref.extractall(self.raw_data_dir)
                    # 重命名为标准格式
                    extracted_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                    if extracted_files:
                        extracted_path = self.raw_data_dir / extracted_files[0]
                        if extracted_path.exists() and extracted_path != csv_path:
                            extracted_path.rename(csv_path)
            except zipfile.BadZipFile:
                logger.error(f"{date_str} 的 ZIP 文件损坏")
                zip_path.unlink(missing_ok=True)
                return False
            
            # 删除 ZIP 文件
            zip_path.unlink(missing_ok=True)
            
            # 验证 CSV 文件是否存在
            if csv_path.exists():
                self.downloaded_files.add(date_str)
                logger.info(f"成功下载并解压 {date_str} 的 GKG 文件")
                return True
            else:
                logger.error(f"{date_str} 解压后 CSV 文件不存在")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"下载 {date_str} 的 GKG 文件时发生网络错误: {e}")
            zip_path.unlink(missing_ok=True)
            return False
        except Exception as e:
            logger.error(f"下载 {date_str} 的 GKG 文件时发生未知错误: {e}")
            zip_path.unlink(missing_ok=True)
            return False
    
    def download(self) -> dict:
        """下载指定日期范围内的所有 GKG 文件"""
        dates = self._get_date_range()
        logger.info(f"开始下载 GKG 数据，日期范围: {self.config.start_date} 到 {self.config.end_date}，共 {len(dates)} 天")
        
        success_count = 0
        fail_count = 0
        
        # 使用线程池并发下载
        with ThreadPoolExecutor(max_workers=self.concurrent_downloads) as executor:
            future_to_date = {
                executor.submit(self._download_file, date): date 
                for date in dates
            }
            
            with tqdm(total=len(dates), desc="下载 GKG 文件") as pbar:
                for future in as_completed(future_to_date):
                    date = future_to_date[future]
                    try:
                        if future.result():
                            success_count += 1
                        else:
                            fail_count += 1
                    except Exception as e:
                        logger.error(f"处理 {date.strftime('%Y%m%d')} 时发生异常: {e}")
                        fail_count += 1
                    finally:
                        pbar.update(1)
        
        logger.info(f"GKG 下载完成: 成功 {success_count} 个，失败 {fail_count} 个")
        
        return {
            "total": len(dates),
            "success": success_count,
            "failed": fail_count
        }

