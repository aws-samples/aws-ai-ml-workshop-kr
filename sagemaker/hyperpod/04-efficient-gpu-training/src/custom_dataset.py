import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import s3fs
import fsspec
import io
from typing import Optional, Dict, List, Union
import logging
import time
from utils.train_utils import print_gpu_utilization, is_main_process, main_process_print

logger = logging.getLogger(__name__)

class StorageConfig:
    """스토리지 설정을 관리하는 클래스"""
    def __init__(
        self,
        storage_type: str = "lustre",  # "local", "s3", "lustre"
        base_path: str = "datasets",
        s3_bucket: Optional[str] = None,
        s3_access_key: Optional[str] = None,
        s3_secret_key: Optional[str] = None,
        s3_endpoint: Optional[str] = None,
        read_buffer_size: int = 8 * 1024 * 1024,  # 8MB
    ):
        self.storage_type = storage_type.lower()
        self.base_path = base_path
        self.s3_bucket = s3_bucket
        self.s3_access_key = s3_access_key
        self.s3_secret_key = s3_secret_key
        self.s3_endpoint = s3_endpoint
        self.read_buffer_size = read_buffer_size
        
        # S3 파일시스템 초기화
        if self.storage_type == "s3":
            self.s3 = s3fs.S3FileSystem(
                #key=s3_access_key,
                #secret=s3_secret_key,
                #endpoint_url=s3_endpoint,
                use_listings_cache=False
            )
        
    def get_full_path(self, filename: str) -> str:
        """파일의 전체 경로를 반환"""
        if self.storage_type == "s3":
            return f"{self.s3_bucket}/{self.base_path}/{filename}"
        return os.path.join(self.base_path, filename)
    
    def list_files(self, pattern: str = "*.csv") -> List[str]:
        """스토리지의 파일 목록을 반환"""
        if self.storage_type == "s3":
            return self.s3.glob(f"{self.s3_bucket}/{self.base_path}/{pattern}")
        return glob.glob(os.path.join(self.base_path, pattern))
    
    def read_file(self, filename: str) -> pd.DataFrame:
        """파일을 읽어서 DataFrame으로 반환"""
        start_time = time.time()
        
        try:
            if self.storage_type == "s3":
                with self.s3.open(self.get_full_path(filename), 'rb') as f:
                    df = pd.read_csv(f)
            else:
                df = pd.read_csv(self.get_full_path(filename))
            elapsed = time.time() - start_time

            #if is_main_process():
            #   main_process_print(f"[{self.storage_type}] File read took {elapsed:.2f}s: {filename}")

            return df
            
        except Exception as e:
            logger.error(f"Error reading file {filename}: {str(e)}")
            print (f"Error reading file {filename}: {str(e)}")
            raise
class SequenceDataset(Dataset):
    """
    스토리지 I/O를 고려한 시퀀스 분류 데이터셋
    
    특징:
    - 여러 스토리지 타입(Local, S3, Lustre) 지원
    - 레이지 로딩으로 메모리 효율성 확보
    - 파일 캐싱으로 반복적인 I/O 최소화
    - 분산 학습 지원
    - CPU 연산 추가로 데이터로더 성능 측정 가능
    """
    def __init__(
        self,
        storage_config: StorageConfig,
        max_seq_length: int = 512,
        use_cache: bool = True,
        cache_size: int = 1000,  # 캐시할 샘플 수
        is_training: bool = True,
        cpu_iterations: int = 2000  # CPU 연산량 조절용 파라미터
    ):
        self.storage_config = storage_config
        self.max_seq_length = max_seq_length
        self.cache_size = cache_size
        self.use_cache = use_cache
        self.is_training = is_training
        self.num_augments = cpu_iterations
        
        # 메타데이터 로드
        #self.metadata = pd.read_csv(
        #    self.storage_config.get_full_path('metadata.csv')
        #)
        self.metadata = self.storage_config.read_file('metadata.csv')
        
        # 전체 샘플 수 계산
        self.total_samples = self.metadata['num_samples'].sum()
        
        # 파일별 시작 인덱스 계산
        self.file_start_indices = np.cumsum(
            [0] + self.metadata['num_samples'].tolist()[:-1]
        )
        
        # 샘플 캐시 초기화
        self.sample_cache = {}
        
        # CPU 연산을 위한 난수 생성기 초기화
        self.rng = np.random.RandomState(42)
        
    def __len__(self) -> int:
        return self.total_samples
    
    def _find_file_index(self, idx: int) -> tuple:
        """주어진 인덱스가 속한 파일과 파일 내 위치를 찾음"""
        file_idx = np.searchsorted(self.file_start_indices, idx, side='right') - 1
        local_idx = idx - self.file_start_indices[file_idx]
        return file_idx, local_idx

    def _apply_cpu_operations(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> tuple:
        """CPU 집약적인 연산 수행"""
        seq_length = len(input_ids)
        
        # 1. N-gram 통계 계산
        ngram_size = 3
        if seq_length >= ngram_size:
            ngrams = np.array([
                hash(tuple(input_ids[i:i+ngram_size])) % 1000
                for i in range(seq_length - ngram_size + 1)
            ])
            ngram_stats = np.bincount(ngrams, minlength=1000)
            
            # N-gram 통계를 바탕으로 input_ids 변형
            for i in range(seq_length - ngram_size + 1):
                if ngram_stats[ngrams[i]] > np.mean(ngram_stats):
                    input_ids[i:i+ngram_size] = np.clip(
                        input_ids[i:i+ngram_size] * 1.1,
                        0,
                        self.max_seq_length - 1
                    )
        
        # 2. Rolling window 연산
        for _ in range(self.num_augments):
            window_size = min(5, seq_length)
            if seq_length >= window_size:
                # Convolution 연산
                rolling_sum = np.convolve(
                    input_ids,
                    np.ones(window_size),
                    mode='valid'
                )
                rolling_mean = rolling_sum / window_size
                
                # 결과를 input_ids에 반영
                valid_length = len(rolling_mean)
                input_ids[:valid_length] = np.clip(
                    input_ids[:valid_length] + np.round(rolling_mean % 3),
                    0,
                    self.max_seq_length - 1
                )
                
                # Attention mask도 업데이트
                attention_mask = (input_ids > 0).astype(np.int64)
        
        # 3. 위치별 가중치 계산 및 적용
        position_weights = np.exp(-np.arange(seq_length) / seq_length)
        weighted_ids = np.round(input_ids * position_weights)
        input_ids = np.clip(weighted_ids, 0, self.max_seq_length - 1)
        
        # 4. 구간별 통계 계산
        for _ in range(self.num_augments):
            chunk_size = max(seq_length // 4, 1)
            for i in range(0, seq_length, chunk_size):
                chunk = input_ids[i:i+chunk_size]
                chunk_mean = np.mean(chunk)
                chunk_std = np.std(chunk) if len(chunk) > 1 else 0
                
                # 통계를 바탕으로 변형
                if chunk_std > 0:
                    normalized_chunk = (chunk - chunk_mean) / chunk_std
                    input_ids[i:i+chunk_size] = np.clip(
                        chunk_mean + normalized_chunk * chunk_std * 0.9,
                        0,
                        self.max_seq_length - 1
                    )
        
        return input_ids.astype(np.int64), attention_mask
    
    def _load_and_parse_sample(self, file_idx: int, local_idx: int) -> Dict[str, torch.Tensor]:
        """파일에서 샘플을 로드하고 파싱"""
        # 캐시 키 생성
        cache_key = f"{file_idx}_{local_idx}"
        
        # 캐시에서 찾기
        if self.use_cache:
            if cache_key in self.sample_cache:
                return self.sample_cache[cache_key]
        
        # 파일 읽기
        filename = self.metadata.iloc[file_idx]['filename']
        df = self.storage_config.read_file(filename)
        row = df.iloc[local_idx]
        
        # 시퀀스 데이터 파싱
        input_ids = np.array([int(x) for x in row['input_ids'].split(',')])
        attention_mask = np.array([int(x) for x in row['attention_mask'].split(',')])
        
        # CPU 연산량 증가를 위한 적용 (학습 시에만)
        if self.is_training:
            input_ids, attention_mask = self._apply_cpu_operations(input_ids, attention_mask)
        
        # 텐서 변환
        sample = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(row['label'], dtype=torch.long)
        }
        
        # 캐시 업데이트
        if len(self.sample_cache) >= self.cache_size:
            # LRU 방식으로 가장 오래된 항목 제거
            self.sample_cache.pop(next(iter(self.sample_cache)))
        self.sample_cache[cache_key] = sample
        
        return sample
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 파일 및 로컬 인덱스 찾기
        file_idx, local_idx = self._find_file_index(idx)
        
        # 샘플 로드 및 반환
        return self._load_and_parse_sample(file_idx, local_idx)

class StorageAwareDataLoader(DataLoader):
    """
    스토리지 특성을 고려한 DataLoader
    
    특징:
    - 스토리지 타입별 최적화된 배치 크기 조정
    - 분산 학습 지원
    - 프리페치와 캐싱 전략 구현
    """
    def __init__(
        self,
        dataset: SequenceDataset,
        prefetch_factor,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        distributed: bool = False,
        persistent_workers: bool = True
    ):
        # persistent_workers 설정
        if num_workers == 0:  persistent_workers = False

        # 분산 학습을 위한 sampler 설정
        sampler = DistributedSampler(dataset) if distributed else None        
        # DataLoader 초기화
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(shuffle and not distributed),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            sampler=sampler,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers
        )
        
        self.storage_type = dataset.storage_config.storage_type
        
    def get_storage_stats(self) -> Dict[str, float]:
        """스토리지 성능 통계 반환"""
        # 실제 구현에서는 스토리지 타입별 성능 지표 수집
        return {
            "avg_read_time": 0.0,
            "throughput": 0.0,
            "cache_hit_rate": len(self.dataset.sample_cache) / len(self.dataset)
        }

class PrefetchDataLoader:
    """
    데이터 프리페치 기능을 지원하는 DataLoader 래퍼 클래스.
    다음 배치를 미리 GPU로 전송하여 I/O와 컴퓨팅을 오버랩합니다.
    """
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        self.iter = None
        self.next_batch = None
        self.stream = torch.cuda.Stream(device=device)

    def __iter__(self):
        self.iter = iter(self.dataloader)
        self.preload()
        return self

    def preload(self):
        try:
            self.next_batch = next(self.iter)
            with torch.cuda.stream(self.stream):
                for k, v in self.next_batch.items():
                    if isinstance(v, torch.Tensor):
                        self.next_batch[k] = v.to(device=self.device, non_blocking=True)
        except StopIteration:
            self.next_batch = None

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration

        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch

    def __len__(self):
        return len(self.dataloader)