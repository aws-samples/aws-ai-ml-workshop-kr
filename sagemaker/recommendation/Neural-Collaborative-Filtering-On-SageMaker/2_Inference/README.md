# 2_Inference

추촌에서는 1_Train 에서 생성하여 S3에 업로드된 모델 아티펙트 (model.tar.gz) 로 모델 서빙을 합니다.
1_Train/1.2.NCF-Train_Local_Script_Mode.ipynb 노트북 가장 하단의  "%store artifact_path" 를 통해서 artifact_path 변수에 S3 의 경로가 저장이 되어 있습니다 .(예: artifact_path: s3://sagemaker-us-east-1-XXXXXXX/pytorch-training-2022-05-27-09-23-31-515/output/model.tar.gz)

# 1. 실행 주요 파일 

- 2.1.NCF-Inference-Scratch.ipynb
    - 세이지 메이커 배포를 로컬 모드와 호스틀 모드를 단계별 실행
    - 추론을 SageMaker Python SDK 및  Boto3 SDK  구현
- 2.2.NCF-Inference-SageMaker.ipynb
    - 세이지 메이커 배포 및 서빙을 전체 실행
    
# (옵션) 추론 도커 이미지 만들고 추론 
이 부분은 옵션 입니다. 하지만 유스케이스에 따라서 사용자 정의의 "모델 서빙 도커 이미지"를 생성할 필요가 있습니다. 이를 위해서 아래의 3개의 노트북은 좋은 길잡이가 됩니다. 추후에 실행을 권장 드립니다.

* sagemaker_inference_container/container-inference/
    * 1.1.Build_Docker.ipynb
    * 2.1.Package_Model_Artifact.ipynb
* 2.3.NCF-Inference-Custom-Docker.ipynb
    * 세이지 메이커 배포 및 서빙을 전체 실행

