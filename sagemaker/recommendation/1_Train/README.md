# 1. 세이지메이커로 NCF 모델 훈련 하기

# 2. 실행 주요 파일 
- #### [중요] 이 워크샵은 ml.g4dn.xlarge, ml.p3.2xlarge, ml.p3.8xlarge, ml.p3.16xlarge 세이지 메이커 노트북 인스턴스의 `conda_python3`에서 테스트 되었습니다.


- 1_Train
    - 0.0.Setup-Environment.ipynb
        - 필요 패키지 설치 및 로컬 모드 세팅    
    - 1.1.NCF-Train-Scratch.ipynb
        - 세이지 메이커 훈련 잡 없이 훈련 코드를 단계별로 노트북에서 실행
    - 1.2.NCF-Train_Local_Script_Mode.ipynb 
        - 세이지 메이커 로컬 모드,호스트 모드로 훈련 
        - 세이지 메이커 Experiment 사용하여 실험 추적        
    - [옵션] 1.3.NCF-Train_Horovod.ipynb
        - 세이지 메이커 호로보드 로컬 모드, 호스트 모드로 훈련 
    - [옵션] 1.4..NCF-Train_SM_DDP.ipynb
        - 세이지 메이커 Data Parallel Training (DDP) 로 로컬 모드, 호스트 모드로 훈련 
        - [중요] ml.p3.16xlarge 이상의 노트북 인스턴스에서 실행이 가능합니다.

# 참고
- 1.3, 1.4 노트북은 옵션이지만, 분산 훈련을 위한 호로보드 및 SageMaker Distributed Data Parallel 를 사용한 분산 훈련을 합니다. 추후에 실행을 권장 드립니다.
