# 멀티 모달 이미지 서치

## 1. 비즈니스 문제 정의
- "텍스트" 혹은 "이미지" 를 제공하고, 벡터 스토어에 저장된 이미지를 서치하는 문제 입니다. 예를 들어서, query_prompt = "drinkware glass" 로 제공을 해도 해당 이미지가 검색이 되고, drinkware glass 의 이미지를 제공해도 검색이 되어야 하는 문제 입니다. 
- 또한 벡터 스토어에 대용량의 이미지 (예: 1 첨만개, 1 억개, 10 억개) 가 저장이 되고, 검색이 빠른 응답속도를 제공을 하고자 합니다.

## 2. 솔루션
- 솔루션은 텍스트 또는 이미지 쿼리를 기반으로 제품을 검색하고 추천하기 위한 Amazon Titan 다중 모드 임베딩 모델 [Amazon Bedrock Titan 모델](https://aws.amazon.com/bedrock/titan)을 사용하여 이미지를 임베딩으로 인코딩하고, OpenSearch Service의 [k-최근접 이웃(KNN) 기능](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/knn.html)을 사용하여 인덱스를 쿼리 합니다


## 3. 사용 데이터 
- [Amazon Berkeley Objects Dataset](https://registry.opendata.aws/amazon-berkeley-objects/)을 사용하고 있습니다. 이 데이터셋은 다국어 메타데이터와 398,212개의 고유 카탈로그 이미지를 포함한 147,702개의 제품 목록으로 구성되어 있습니다. 8,222개의 목록에는 턴테이블 사진이 포함되어 있습니다. 여기서는 제품 이미지와 영어로 된 제품 이름(제품의 간단한 설명으로 간주)만 사용할 것입니다. 데모를 위해 약 1,600개의 제품을 사용할 예정입니다.


## 4.실험 환경
### 4.1 SageMaker Studio Code Editor
- 노트북은 [SageMaker Studio Code Editor](https://docs.aws.amazon.com/sagemaker/latest/dg/code-editor.html) 및 커널 base (Python 3.10.13) 에서 테스트 되었습니다.

### 4.2 기타 환경
**요구 사항**
- AWS OpenSearch 2.1 이상 
    - 아래 샘플 노트북 1번에서 설치 합니다.


## 5.실행 노트북
- 아래를 단계별로 실행 하세요.
    - notebook/01_setup_opensearch_simple.ipynb
    - notebook/02_multi_modal_image_search.ipynb
    - notebook/03_large_scale_multi_modal_image_search.ipynb