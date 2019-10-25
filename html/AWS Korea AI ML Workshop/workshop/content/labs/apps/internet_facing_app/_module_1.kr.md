+++
title = "Module 1: Notebook Instance 생성하기 "
menuTitle = "Notebook Instance 생성하기"
date = 2019-10-23T15:44:53+09:00
weight = 312
+++

### S3 Bucket생성하기 ###

SageMaker는 S3를 데이터와 모델 저장소로 사용합니다. 여기서는 해당
목적으로 S3 Bucket을 생성합니다.

1\) AWS 관리 콘솔 (<https://console.aws.amazon.com/>)에 Sign in 합니다.

2\) AWS Services 리스트에서 S3 로 이동합니다.

3\) `+ Create Bucket` 버튼을 선택합니다.

4\) 아래 내용 설정 후 화면 왼쪽 아래 `Create` 클릭합니다.

-   Bucket name: sagemaker-{userid} \[반드시 고유한 값 설정\]

-   Region : Asia Pacific (Seoul)

![](/images/apps/internet_facing_app/image2.png)

### Notebook instance 생성 ###

1\) AWS관리 콘솔에서 오른쪽 상단에서 Seoul Region선택 후 AWS Services
리스트에서 Amazon SageMaker 서비스를 선택합니다.

![](/images/apps/internet_facing_app/image3.png)

2\) 새로운 Notebook instance를 생성하기 위해 왼쪽 패널 메뉴 중 `Notebook Instances` 선택 후 오른쪽 상단의 `Create notebook instance` 버튼을
클릭 합니다.

![](/images/apps/internet_facing_app/image4.png?width="5.7011187664042in"
height="2.9270352143482063in)

3\) Notebook instance 이름으로 `\[First Name\]-\[Last Name\]-workshop`으로 넣은 뒤 `ml.m4.xlarge` 인스턴스 타입을 선택
합니다.

![](/images/apps/internet_facing_app/image5.png?width="4.330936132983377in"
height="5.555683508311461in)

4\) IAM role은 `Create a new role`을 선택하고, 생성된 팝업창에서는
`S3 buckets you specify -- optional` 밑에 `Specific S3 Bucket`을 선택
합니다. 그리고 텍스트 필드에 위에서 만든 S3 bucket 이름(예:
sagemaker-xxxxx)을 선택 합니다. 이후 `Create role`을 클릭합니다.

![](/images/apps/internet_facing_app/image6.png?width="5.361015966754156in"
height="4.625900043744532in)

5\) 다시 Create Notebook instance 페이지로 돌아온 뒤 `Create notebook`
`instance`를 클릭합니다.

### Notebook Instance 접근하기 ###

1\) 서버 상태가 `InService` 로 바뀔 때까지 기다립니다. 보통 5분정도의
시간이 소요 됩니다.

![](/images/apps/internet_facing_app/image7.png?width="6.988888888888889in"
height="1.9180555555555556in)

2\) `Open`을 클릭하면 방금 생성한 notebook instance의 Jupyter 홈페이지로
이동하게 됩니다.

![](/images/apps/internet_facing_app/image8.png?width="6.988888888888889in"
height="1.9534722222222223in)