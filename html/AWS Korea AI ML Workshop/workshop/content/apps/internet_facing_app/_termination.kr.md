+++
title = "서비스 종료 가이드"
date = 2019-10-23T15:49:29+09:00
weight = 399
+++

워크 샵 이후 발생 되는 비용을 방지하기 위해서 아래의 단계에 따라 모두
종료/삭제 해 주세요. 비용이 발생하더라도 실습하신 Internet-facing App을
유지하고 싶으신 경우에는 아래의 Notebook instance의 경우만 처리하시면
됩니다.

### Notebook instance

만약 향후 사용을 위해 인스턴스를 저장하고 싶다면 **stop**을 하시면 됩니다. 이 경우 스토리지 비용은 발생합니다. 향후 다시 재가동 하시려면 Start button을 클릭하면 됩니다.

![](/images/apps/internet_facing_app/image61.png?width="6.39090113735783in"
height="1.6953641732283464in)

<center>**Figure 50. SageMaker 노트북 인스턴스 중단 화면.**</center>

삭제를 할 경우는 **stop** 되어 있는 해당 notebook instance를 선택하고 **Action** Dropdown 메뉴에서 **Delete** 선택 하시면 됩니다.

![](/images/apps/internet_facing_app/image62.png?width="6.43708552055993in"
height="1.7037642169728784in)

<center>**Figure 51. SageMaker 노트북 인스턴스 삭제 화면.**</center>

### SageMaker Endpoints

훈련된 모델을 실제 예측 업무를 위해 배포된 한대 이상으로 구성된
클러스터입니다. Notebook안에서 명령어로 삭제하거나 SageMaker console에서
삭제 하실 수 있습니다. 삭제 하시기 위해서는 왼쪽 패널의 Endpoints를 선택
하신 후 해당 endpoints들 옆에 radio button을 클릭 하신 후 Action
Dropdown 메뉴에서 Delete 선택 하시면 됩니다.

![](/images/apps/internet_facing_app/image63.png?width="6.0957469378827644in"
height="2.28166447944007in)

<center>**Figure 52. SageMaker Endpoint 삭제 화면.**</center>

### Lambda instance: 생성하신 Lambda instance를 삭제합니다.

![](/images/apps/internet_facing_app/image64.png?width="5.866441382327209in"
height="1.5744499125109361in)

<center>**Figure 53. Lambda 인스턴스 삭제 화면.**</center>

### Amazon API Gateway instance: 생성하신 Gateway instance를 삭제합니다.

![](/images/apps/internet_facing_app/image65.png?width="4.668955599300087in"
height="2.4699453193350833in)

<center>**Figure 54. API Gateway 삭제 화면.**</center>

### Amazon S3 buckets: 생성하신 S3 Bucket (SageMaker용, Public Internet용)들을 모두 삭제합니다.

![](/images/apps/internet_facing_app/image66.png?width="6.065811461067367in"
height="3.493386920384952in)
<center>**Figure 55. S3 버킷 삭제 화면.**</center>

{{% notice info %}}
이상으로 본 핸즈온 세션의 모든 과정을 마무리 하셨습니다. 수고하셨습니다.
{{% /notice %}}