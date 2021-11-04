+++
title = "Module 4: SageMaker Endpoint 호출 Lambda 함수 개발하기"
menuTItle = "SageMaker Endpoint 호출 Lambda 함수 개발하기"
date = 2019-10-23T15:50:41+09:00
weight = 315
+++

본 모듈에서는 방금 생성한 SageMaker의 Inference service를 호출하는
Lambda 함수를 개발해 보겠습니다.

### Lambda 함수 생성하기 

1. AWS 콘솔에서 Lambda를 선택 (<https://console.aws.amazon.com/lambda>)

1. `Create function` 선택 (Figure 15 참조)

    ![](/images/apps/internet_facing_app/image26.png?width="4.7558814523184605in"
    height="0.8586482939632546in)

    <center>**Figure 15. Lambda 함수 생성 화면.**</center>

    ![](/images/apps/internet_facing_app/image27.png?width="5.687501093613299in"
    height="4.465117016622922in)

    <center>**Figure 16. Lambda 함수 생성 화면.**</center>

1. Lambda 생성화면에서 Figure 16과 같이 Lambda 함수 이름과 Runtime
    (Python 3.6) 그리고 Role은 `Create a custom role`을 선택합니다.

    1. Name : MySeq2SeqInference으로 지정.

    1. Runtime: Python 3.6으로 지정

    1. `Role: Create a custom role`을 선택하면 Figure 17와 같이 `AWS
        Lambda required access to your resources`가 나옵니다. 여기서
        `Allow` 를 누릅니다.

    1. Allow 클릭하면 창이 닫히고 Lambda Console로 돌아가는 데 여기서
        Create Function을 선택하시면 됩니다.

    ![image28](/images/apps/internet_facing_app/image28.png?width="4.660433070866142in"
    height="4.294642388451444in)

    <center>**Figure 17. AWS Lambda 접근 허락 화면.**</center>

### Lambda 함수에 Role을 추가하기 

방금 생성한 Lambda 함수에 새롭게 추가된 Role에 SageMaker와 API Gateway를
사용할 수 있는 정책 (Policy)를 추가해보겠습니다.

1. AWS 콘솔에서 IAM 서비스를 선택하세요.

1. 왼편의 메뉴에서 `Roles`를 클릭하세요.

1. 방금 생성하신 Lambda에 사용되는 Role을 선택하세요 (Figure 18 참조)

    ![image29](/images/apps/internet_facing_app/image29.png?width="6.761539807524059in"
    height="3.4069761592300964in)

    <center>**Figure 18. Lambda 함수 선택.**</center>

1. "Add inline policy"를 선택하세요 (Figure 19 참조).

    ![](/images/apps/internet_facing_app/image30.png?width="6.461656824146981in"
    height="3.686046587926509in)

    <center>**Figure 19. IAM Role에 정책을 추가하는 화면.**</center>

1. 다음 화면의 검색창에 `SageMaker` 입력 하세요 (Figure 20 참조).

    ![](/images/apps/internet_facing_app/image31.png?width="5.719290244969379in"
    height="3.301205161854768in)

    <center>**Figure 20. AmazonSageMakerFullAccess 정책 추가 화면.**</center>

1. Access level at Actions에 있는 모든 `DescribeEndpoint` and
    `InvokeEndpoint` 를 선택하세요 (See Figure 21).

    ![](/images/apps/internet_facing_app/image32.png?width="5.534816272965879in"
    height="4.030120297462817in)

    <center>**Figure 21. Select DescribeEndpoint and InvokeEndpoint in the Access
    level.**</center>

1. 하면 하단의 Resources에 있는 노란색의 `You chose actions that
    require the endpoint-config resource type` 문장을 선택하신 후
    Figure 22 화면과 같이 Resources 섹션에 있는 `Any` 를 선택합니다.
    이후 화면 하단에 있는 `Review policy`를 선택하세요.

    ![](/images/apps/internet_facing_app/image33.png?width="5.009854549431321in"
    height="3.7409634733158357in)

    <center>**Figure 22. Select endpoint resource type.**</center>

1. `Review policy` 다이얼로그에서 새로운 policy 이름을 입력하신 후 화면
    하단의 Create policy버튼을 선택하세요 (See Figure 30).

    ![](/images/apps/internet_facing_app/image34.png?width="5.000937226596675in"
    height="2.5372397200349956in)

    <center>**Figure 23. Create policy screen.**</center>

1. 최종 추가된 Policy가 그림 19와 동일한지 확인

    ![](/images/apps/internet_facing_app/image35.png?width="6.229643482064742in"
    height="3.4329899387576552in)

    <center>**Figure 24. 최종 Role의 정책들 화면.**</center>

### Lambda 함수 코딩하기

다시 AWS 콘솔의 Lambda 서비스 화면으로 이동하신 후 윗 단계에서 생성하신
Lambda를 선택합니다. Figure 25 과 같이 추가된 Role의 Policy들을 확인하실
수 있습니다.

![](/images/apps/internet_facing_app/image36.png?width="5.166319991251093in"
height="3.4060542432195975in)

<center>**Figure 25. Lambda 선택 화면.**</center>

현 페이지에서 마우스를 스크롤해서 하단으로 이동하면 Figure 26와 같이
Lambda의 내장 코드들을 직접 수정할 수 있는 인터페이스가 제공이 됩니다.

![](/images/apps/internet_facing_app/image37.png?width="5.16226924759405in"
height="3.252577646544182in)

<center>**Figure 26. Lamba 코드 개발 화면.**</center>

AWS Lambda는 AWS 콘솔 상에서 바로 코딩할 수 있게 Cloud9 에디터가 내장되어
있습니다. 아래의 순서에 따라 Lambda 함수를 만들어 보겠습니다.

1. 다음 페이지의 Python 샘플 코드를 `Copy` 후 `Paste` 로 Lambda의 online
    editor에 입력합니다. Python 코드를 복사 및 붙여 넣기를 할때는 원
    코드의 indent를 그대로 지키는 것이 중요합니다. 현재 보시고 있는 PDF
    문서 상에서 복사가 제대로 되지 않는 경우 아래 온라인 주소에서
    소스코드를 복사하 셔도 됩니다:

    https://raw.githubusercontent.com/aws-samples/aws-ai-ml-workshop-kr/master/src/release/2018-11/lambda_function.py


1. 붙여넣기 하신 소스코드 상의 `endpoint_name` 을 본 실습 동안 생성한 Seq2Seq endpoint 서버 주소로 변경하십시요 (Figure 27 참조).

    ![](/images/apps/internet_facing_app/image38.png?width="5.284501312335958in"
    height="1.3815102799650043in)

    <center>**Figure 27. SageMaker EndPoint 이름 확인 방법.**</center>

**Labmda Python sample Code**

1.  Endpoint용으로 선택하신 서버의 Instance Type과 번역을 하기위한
    text의 크기에 따라 번역에 몇초 이상이 소요될 수도 있습니다. 이
    시간동안 Lambda 함수 호출이 Timeout 되는 것을 방지하기 위해 Figure
    28와 같이 Lambda의 Timeout 시간을 10초로 늘입니다.

1.  상단의 `Save` 버튼을 눌러 저장합니다.

    ![](/images/apps/internet_facing_app/image39.png?width="2.499987970253718in"
    height="2.046392169728784in)

    <center>**Figure 28. Lambda 함수 Timeout 값 조정.**</center>

새로 만든 Lambda 함수의 동작을 바로 확인할 수 있습니다.

1. Figure 29와 같이 `Configure test events`를 선택합니다.

    ![](/images/apps/internet_facing_app/image40.png?width="3.3404232283464568in"
    height="0.776233595800525in)

    <center>**Figure 29. Lambda 테스트 데이터 구성 화면.**</center>

1. Event name을 입력합니다 (예: SampleEnglishSentence)

1. 하단의 테스트 이벤트 입력화면에서 Figure 30과 같이 아래의 샘플 영어
    문장을 입력합니다. 또는
    <https://raw.githubusercontent.com/pilhokim/ai-ml-workshop/master/2018-09/sample_query.json>
    에서 복사해서 사용하셔도 됩니다.

    ![](/images/apps/internet_facing_app/image41.png?width="6.3796237970253715in"
    height="3.011049868766404in)

    <center>**Figure 30. Test 이벤트 생성.**</center>

    이때 주의하실점은 JSON 형식의 `sentences`와 `query`는 미리 약속된 key
    값이므로 변경을 하시면 안됩니다.

1. Create 버튼을 선택합니다.

1. 입력이 완료 된 후 상단의 `Test` 버튼을 클릭하시면 Figure 31와 같은
    화면이 보이면 정상적으로 작동하는 것을 확인하실 수 있습니다. 하단의
    Cloud9에서도 결과를 확인하실 수 있습니다.

    ![](/images/apps/internet_facing_app/image42.png?width="6.027586395450569in"
    height="3.238391294838145in)

    <center>**Figure 31. Lambda 함수 테스트 결과 화면.**</center>