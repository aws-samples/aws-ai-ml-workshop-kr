import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def generate_sequence_data(output_dir, num_files=1000, seq_len=512, vocab_size=30000):
    """
    시퀀스 분류를 위한 대용량 샘플 데이터셋을 생성합니다.
    
    Parameters:
    -----------
    output_dir : str
        데이터를 저장할 디렉토리 경로
    num_files : int
        생성할 파일 개수
    seq_len : int
        각 시퀀스의 길이
    vocab_size : int
        어휘 사전 크기
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 메타데이터 저장을 위한 리스트
    metadata = []
    
    for file_idx in tqdm(range(num_files), desc="Generating files"):
        # 각 파일별 데이터 생성
        num_samples = np.random.randint(100, 1000)  # 파일당 100-1000개의 샘플
        
        data = {
            'sequence_id': list(range(num_samples)),
            'input_ids': [
                ','.join(map(str, np.random.randint(100, vocab_size, seq_len)))
                for _ in range(num_samples)
            ],
            'attention_mask': [
                ','.join(map(str, np.random.randint(0, 2, seq_len)))
                for _ in range(num_samples)
            ],
            'label': np.random.randint(0, 2, num_samples)
        }
        
        # DataFrame 생성 및 저장
        df = pd.DataFrame(data)
        filename = f'sequence_data_{file_idx:05d}.csv'
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        
        # 메타데이터 추가
        metadata.append({
            'filename': filename,
            'num_samples': num_samples,
            'file_size_bytes': os.path.getsize(filepath)
        })
    
    # 메타데이터 저장
    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
    
    # 데이터셋 통계 출력
    total_samples = meta_df['num_samples'].sum()
    total_size_gb = meta_df['file_size_bytes'].sum() / (1024**3)
    
    print(f"\nDataset Generation Complete!")
    print(f"Total files: {num_files}")
    print(f"Total samples: {total_samples:,}")
    print(f"Total size: {total_size_gb:.2f} GB")
    print(f"Average samples per file: {total_samples/num_files:.1f}")

if __name__ == "__main__":
    # 샘플 데이터셋 생성
    generate_sequence_data(
        output_dir="datasets",
        num_files=1000,  # 1000개 파일
        seq_len=512,     # 시퀀스 길이
        vocab_size=30000 # 어휘 크기
    )