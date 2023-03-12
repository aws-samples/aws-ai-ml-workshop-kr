# Trn1 & INF2 

## 1. 실습 파일 구성

### 워밍업
- warming-up/compile_getting_started.ipynb

### NCF 모델 컴피알 및 추론 스크래치
- 0.0.Setup_Environment.ipynb
    - 환경 설정
- 1.1.NCF-on-INF2.ipynb
    - NCF 모델 컴파일 및 추론  
- 2.1.NCF-on-INF2-Benchmark.ipynb    
    - NCF 모델 벤치 마킹


## 2. 벤치 마킹 실험
### 스펙: instqance = ml.p3.2xlarge (GPU - V100 1장)
- throughput = inferences / duration
- inferences : 인퍼런스 개수
- duration : 벤치 마크가 총 수행된 초 (단위: second)
- Latency: P50, P95, P99 - 전체 추론 시간의 퍼센타일 정보
    - (단위: milli-second)
- 참조
    - 2_Inference/2.4.NCF-Inference-benchmark.ipynb
 
 ```
Batch Size:  1
Batches:     1000
Inferences:  1000
Threads:     1
Models:      1
Duration:    0.594
Throughput:  1682.111
Latency P50: 0.585
Latency P95: 0.624
Latency P99: 0.637
 ```
  
### 스펙: instqance = inf2.2xlarge (Neurn Core: 2)
- 참조
    - 2.1.NCF-on-INF2-Benchmark.ipynb
 
```
Filename:    models/model.pt
Batch Size:  1
Batches:     2000
Inferences:  2000
Threads:     2
Models:      2
Duration:    0.115
Throughput:  17406.568
Latency P50: 0.111
Latency P95: 0.125
Latency P99: 0.135
```

### 결론
inf2.2xlarge 의 추론 속도는 p50으로 볼때 `Latency P50: 0.111 (단위: 밀리 세컨드)` ml.p3.2xlarge는  `Latency P50: 0.585` 으로 약 5배의 차이가 발생 함. 