import torch
import torch.distributed as dist

from pynvml import *

def print_gpu_utilization():
    """GPU 메모리 사용량 통계 출력"""
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used//1024**2  # MB 단위로 반환


def is_main_process():
    """현재 프로세스가 메인 프로세스(global_rank=0)인지 확인합니다."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def main_process_print(*args, **kwargs):
    """메인 프로세스(global_rank=0)에서만 출력합니다."""
    if is_main_process():
        print(*args, **kwargs)

# Prefetch를 지원하는 DataLoader 래퍼 클래스
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