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

