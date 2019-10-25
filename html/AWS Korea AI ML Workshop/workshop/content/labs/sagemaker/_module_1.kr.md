+++
title = "Module 1: S3 bucket과 Notebook Instance 생성하기"
menuTitle = "S3 bucket과 Notebook Instance 생성하기"
date = 2019-10-15T15:16:07+09:00
weight = 201
+++

## S3 Bucket 생성하기 

SageMaker는 S3를 데이터와 모델 저장소로 사용합니다. 여기서는 해당 목적으로 S3 Bucket을 생성합니다. 오늘 실습에서는 `N. Virginia (us-east-1)` 리전을 사용합니다.

1. [AWS 관리 콘솔](https://console.aws.amazon.com/)에 Sign in 합니다: 
    {{% button href="https://console.aws.amazon.com/" icon="fas fa-terminal" %}}Open AWS Console{{% /button %}}
    {{% notice info %}}
    만약 AWS 측에서 Event Engine을 사용하여 임시 아이디를 생성한 경우 제공받으신 URL을 여시고 team hash code를 입력하시면 됩니다.
    {{% /notice %}}
1. AWS Services 리스트에서 S3 로 이동합니다.
1. `"+ Create Bucket"` 버튼을 선택합니다.
1. 아래 내용 설정 후 화면 왼쪽 아래 Create 클릭합니다.

* Bucket name: sagemaker-{userid}  [반드시 고유한 값 설정] 
* Region : US East (N. Virginia)

![create_s3_bucket](/images/sagemaker/module_1/create_s3_bucket.png?classes=border)

## Notebook instance 생성

1. AWS관리 콘솔에서 오른쪽 상단에서Region선택 후 AWS Services 리스트에서 Amazon SageMaker 서비스를 선택합니다.
    ![aws_console_sagemaker_selection](/images/sagemaker/module_1/aws_console_sagemaker_selection.png)

1. 새로운 Notebook instance를 생성하기 위해 왼쪽 패널 메뉴 중 Notebook Instances 선택 후 오른쪽 상단의 `Create notebook instance` 버튼을 클릭 합니다.

    ![sagemaker_create_notebook_instance](/images/sagemaker/module_1/sagemaker_create_notebook_instance.png)

1. Notebook instance 이름으로 `[First Name]-[Last Name]-workshop` 으로 넣은 뒤 `ml.m4.xlarge` 인스턴스 타입을 선택 합니다. 

    ![sagemaker_notebook_instance_setting](/images/sagemaker/module_1/sagemaker_notebook_instance_setting.png)

1. IAM role은 `Create a new role` 을 선택하고, 생성된 팝업 창에서는 `S3 buckets you specify – optional` 밑에 `Specific S3 Bucket` 을 선택 합니다. 그리고 텍스트 필드에 위에서 만든 S3 bucket 이름(예: sagemaker-xxxxx)을 선택 합니다. 이후 `Create role` 을 클릭합니다.

    ![sagemaker_create_an_iam_role](/images/sagemaker/module_1/sagemaker_create_an_iam_role.png)

1. 다시 Create Notebook instance 페이지로 돌아온 뒤 `Create notebook instance` 를 클릭합니다.

## Notebook Instance 접근하기

1. 서버 상태가 `InService` 로 바뀔 때까지 기다립니다. 보통 5분정도의 시간이 소요 됩니다.

    ![sagemaker_instance_status](/images/sagemaker/module_1/sagemaker_instance_status.png)

1. `Open Jupyter`를 클릭하면 방금 생성한 notebook instance의 Jupyter 홈페이지로 이동하게 됩니다.

    ![sagemaker_new_jupyter_notebook](/images/sagemaker/module_1/sagemaker_new_jupyter_notebook.png)

수고하셨습니다. 모듈 1을 완료하였습니다.