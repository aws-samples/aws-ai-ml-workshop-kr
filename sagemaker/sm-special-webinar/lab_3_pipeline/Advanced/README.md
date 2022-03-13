# 고급 SageMaker-Pipelines-Step-By-Step 워크샵

## 1.개요
lab_3_pipeline 버전에서 기본적인 데이터 전처리, 모델 훈련, 모델 배포 및 추론을 위한 개념 및 파이프라인 단계를 배웠습니다. 이제 조금더 다양한 파이프라인 단계 및 개념에 대해서 다루고자 합니다. 고급 과정에서는 다음과 같은 파이프라인 단계를 언급합니다. 개발자 가이드의 상세 단계는 여기를 참조 하세요.
- [파이프라인 단계 유형](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/build-and-manage-steps.html#build-and-manage-steps-types) 

---


## 2. 워크샵 실행 후 결과
이 워크샵은 1개의 파이프라인을 생성합니다. 이후에 실행은 2번 합니다. 
- 첫번째는 현재 정의된 파이프라인을 실행하고, 조건 스텝에서 if 구문의 True 가 발생한 흐름으로 실행 합니다.
- 두번째는 파라미터로 valuation:roc-auc 의 평가 지표값을 바꾸어서 조건 스템에서 if 구문의 False 가 발생한 흐름으로 실행 합니다.
    - 파라미터 기반 실행을 진행 하며, 캐시를 이용한 실행을 합니다.

### (1) 첫 번째 파이프라인 실행 후 결과 (약 16분 소요)
- `데이터 전처리 --> 모델 훈련 --> 모델 평가 --> 모델 평가 지표의 조건 문 --> 하이퍼 파라미터 튜닝` 의 단계를 진행 함.

![hpo-pipeline.png](img/hpo-pipeline.png)

### (2) 파라미터 입력을 통한 파이프라인 실행 후 결과 (캐싱 이용, 10초 이내 소요됨.)
- `데이터 전처리 (캐시이용) --> 모델 훈련 (캐시 이푱) --> 모델 평가 (캐시 이용)  --> 모델 평가 지표의 조건 문 --> 모델 레지스트리 등록 / 세이지 메이커 모델 생성` 의 단계를 진행 함.


![cache-pipeline-result.png](img/cache-pipeline-result.png)


## 3.고급 단계에서 사용하는 단계 들
- Processing
    - 데이터 세트의 전처리 단계
    - 훈련 모델의 성능 평가 단계
- Training
    - XGBoost 알로리즘 훈련 단계
- Condition
    - 모델 평가의 기반하에 Validation:roc-auc 값의 따른 if-else 단계 
- RegisterModel    
    - 훈련 모델을 모델 레지스트리에 등록 단계
- Tuning
    - XGBosst 의 하이퍼파라미터 튜닝 단계
- CreateModel
    - 모델 배포를 위한 세이지 메이커 모델 단계

## 4.핸즈온 환경
- [필수] 세이지 메이커의 노트북 인스턴스에서 쥬피터 노트북을 통해 실행하세요.
- [권장] 또한 세이지 메이커의 스튜디오에서 생성된 파이프라인을 GUI 로 보면서 상태를 확인 하세요.


## 5. 노트북 구성

### [Quick Approach] 
Step-By-Step으로 접근하지 않고, 빠르게 보시기 위해서는 아래 노트북만을 수행 하세요.
- 0.0.Setup-Environment.ipynb
- 1.1.Prepare-Dataset.ipynb
- 5.1.All-Pipeline.ipynb

### [Step-By-Step 접근]

- 0.0.Setup-Environment.ipynb
    - 필요한 파이썬 패키지를 설치 합니다.
    
    
- 1.1.Prepare-Dataset.ipynb
    - 데이터 세트 준비를 하고 S3에 업로드 합니다.


- 2.1.Train-Exp-Pipeline.ipynb
    - 아래의 세가지 방법 이외 (Scratch 에 포함)에 **Experiment(실험)** 를 추가하여, Trial(시도)에 따른 모델 버전 들을 추적 함.
    - 또한 훈련된 모델을 **모델 레지스트"에 등록까지 하는 과정 포함
        - 로컬 노트북에서 전처리 스크립트 실행 (예: python preprocess.py)
        - 로컬 노트북에서 로컬 모드의 다커 컨테이너로 실행
        - MBP 의 파이프라인 단계 (step_proecess) 를 생성하여 실행


- 3.1.HPO-Pipeline.ipynb
    - HPO (Hyperparameter Optimization) 를 사용 및 파이프라인의 HPO 단계를 생성
    
    
- 4.1.Eval-Pipeline.ipynb
    - 훈련된 모델을 테스트 데이터 세트로 평가하여 모델 지표를 생성하고, 이를 **조건 단계** 를 통해서 if-else 문 처리를 함.
    
    
- 5.1.All-Pipeline.ipynb
    - 위의 과정을 모두 연결하여 하나의 파이프 라인으로 생성.



## [참고 자료]
- 추후 작성 예정





---

