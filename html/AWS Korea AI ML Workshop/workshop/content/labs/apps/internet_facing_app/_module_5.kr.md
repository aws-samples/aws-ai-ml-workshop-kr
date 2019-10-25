+++
title = "Module 5: AWS API Gateway와 S3 Static Web Server를 이용한 웹서비스 연결하기"
menuTitle = "AWS API Gateway와 S3 Static Web Server를 이용한 웹서비스 연결하기"
date = 2019-10-23T15:51:09+09:00
weight = 316
+++

### API Gateway 생성 및 Lambda함수 연결하기

1. Amazon API Gateway 콘솔 접속
    (<https://console.aws.amazon.com/apigateway/> )

1.  "Create API" -> "New API" 선택

1.  셋팅에서 새로운 API name 입력 (ex.
    `SageMakerSeq2SeqLambdaGateWay`)후 `Endpoint Type`을 `Regional` 로
    선택 (Figure 32 참조).

    ![](/images/apps/internet_facing_app/image43.png?width="5.980104986876641in"
    height="3.0316469816272966in)

    <center>**Figure 32. Amazon API Gateway 생성 화면.**</center>

1.  바뀐 화면에서 `Actions` -> `Create Method` 선택

1.  하단의 콤보 박스에서 `POST` 선택 (Figure 33 참조)

1. 체크(`V`) 버튼 클릭해서 적용 (Figure 33 참조)

    ![file:///var/folders/j6/t\_\_2vd7n2070g\_gysxbr7nt04v998b/T/com.microsoft.Word/screenshot.png](/images/apps/internet_facing_app/image44.png?width="3.675680227471566in"
    height="1.1166622922134732in)

    <center>**Figure 33. POST method 추가 화면.**</center>

1.  오른편의 셋업에서 아래와 같이 입력 진행 (Figure 34 참조)

    1.  Integration type: `Lambda function`

    1.  Lambda region: Labmda를 생성하신 Region (`us-east-1`) 입력

    1.  Lambda function: Lambda 함수 이름 입력

    1.  `Save` 선택

![](/images/apps/internet_facing_app/image45.png?width="6.243467847769029in"
height="2.6967760279965005in)

<center>**Figure 34. Lambda 함수를 호출하기 위한 Gateway POST method 셋팅 화면.**</center>

API Gateway가 생성이 된 이후에는 Figure 35와 같이 Test를 진행하여 제대로
Lambda를 호출하는지 확인하실 수 있습니다.

1.  `Test`를 선택하셔서 API Gateway의 testing interface를 확인합니다.

1.  Request body에 Lambda 호출에 사용되었던 아래의 예제 데이터를
    입력하신 후 `Test`를 선택합니다.

    ![](/images/apps/internet_facing_app/image46.png?width="6.871837270341207in"
    height="3.517856517935258in)

    <center>**Figure 35. API Gateway Test 화면**</center>

    테스트 결과가 Figure 36과 같이 보이면 정상적으로 동작하는 것으로
    확인하실 수 있습니다.

    ![](/images/apps/internet_facing_app/image47.png?width="6.882856517935258in"
    height="3.3265310586176726in)

    <center>**Figure 36. API Gateway 테스트 결과.**</center>

1.  `Enable CORS: S3 Static Web Server`를 이용해서 API Gateway를 호출하면
    origin이 다르기 때문에 반드시
    [CORS](https://en.wikipedia.org/wiki/Cross-origin_resource_sharing)
    (Cross-Origin Resource Sharing)를 Enable 해야만 외부 싸이트에서 이
    REST 서비스를 이용할 수 있게 됩니다.

  * `Actions` -> `Enable CORS` 선택 (Figure 37 참조)

    ![](/images/apps/internet_facing_app/image48.png?width="6.737308617672791in"
    height="2.2767858705161856in)

    <center>**Figure 37. API Gateway API Enable CORS 화면.**</center>

  *  `Enable CORS and replace existing CORS headers` 선택

  *  `Yes, replace existing values` 선택 (Figure 38 참조)

    ![](/images/apps/internet_facing_app/image49.png?width="6.457083333333333in"
    height="2.446428258967629in)

    <center>**Figure 38. CORS replace existing values 화면.**</center>

1.  정상적으로 동작이 되면 `Actions`->`Deploy API` 선택 (Figure 39 참조)
    합니다. API Deploy를 반드시 하셔야 실제 외부 (Public Internet)에서
    호출을 할 수 있습니다.

1. 현재 생성한 Gateway의 stage 이름을 부여합니다. 예제에서는 "pr`od"라는
    약어로 stage 이름을 정의하였습니다. 개발 단계에 따라 `test` 나
    `prod` 등 의미 있는 키워드를 부여하시면 됩니다.

    ![](/images/apps/internet_facing_app/image50.png?width="6.574346019247594in"
    height="2.401360454943132in)

    <center>**Figure 39. API deploy 화면.**</center>

1. Deploy가 된 이후 Stage Editor에서 invoke URL을 (Figure 40 참조)
    메모장에 따로 기록해 두시고 `SDK Generation` -> P`latform
    (JavaScript)` -> `Generate SDK` 선택. 이 JavaScript 라이브러리는 API
    Gateway 서비스에 대해
    [CORS](https://en.wikipedia.org/wiki/Cross-origin_resource_sharing)
    (Cross-Origin Resource Sharing)을 지원해주는 기능을 포함하고
    있습니다.

    ![](/images/apps/internet_facing_app/image51.png?width="5.9818952318460195in"
    height="2.03576990376203in)

    <center>**Figure 40. API Gateway 접속 SDK 다운로드 화면**</center>

이제 S3를 이용해서 static web server를 설정하기 위한 화일들을
준비하겠습니다.

* 상기 API Gateway SDK 생성으로 다운 받은 압축 화일을 임의의
디렉토리에 푸세요 (unzip).

*  S3 Static 웹 서버에 사용될 index.html 과 error.html 파일을 다음의 S3
버켓에서 다운로드 하여 상기 단계에서 사용된 디렉토리에 동일하게
저장합니다:
<https://s3.amazonaws.com/pilho-sagemaker-ai-workshop-lambda/index_error_html.zip>

*  최종 파일들이 Figure 41과 같이 구성되어 있으면 됩니다. 이 파일들은
다음 단계에서 만들 S3 버킷에 업로드 되게 됩니다.

![](/images/apps/internet_facing_app/image52.png?width="6.461221566054244in"
height="0.9090901137357831in)

<center>**Figure 41. 웹서버 구성 화일 리스트 화면.**</center>

### S3 Static Web Server 생성하기

1.  Amazon S3 콘솔 접속 (<https://s3.console.aws.amazon.com> )

1.  `Create bucket` 선택

1.  새로운 버킷 이름 입력 (ex. `jihye-sagemaker-public-test`) -> `Next`
    -> `Next` 선택

1.  Set permissions에서 `Manage public permissons`를 `Grant public read
    access to this bucket` 으로 설정 (Figure 43 참조)

    ![file:///var/folders/j6/t\_\_2vd7n2070g\_gysxbr7nt04v998b/T/com.microsoft.Word/screenshot.png](/images/apps/internet_facing_app/image53.png?width="4.00056539807524in"
    height="4.580357611548556in)

    <center>**Figure 42. S3 버킷 Public 접속 허용 화면.**</center>

1.  `Next`->`Create bucket` 선택

1.  생성된 S3 bucket 선택

1.  `Properties` -> `Static website hosting` -> `Use this bucket to
    host a website` 선택 후 `Index document : index.html`, `Error document
    : error.html` 입력

1.  `Save` 선택 (Figure 43 참조)

1.  이 단계 까지 마치신 후 상단의 URL 형식의 Endpoint 주소를 기록해
    둡니다. 이 URL 주소를 이용해서 S3 웹 서버에 접속하게 됩니다.

    ![](/images/apps/internet_facing_app/image54.png?width="5.493757655293089in"
    height="3.375734908136483in)

    <center>**Figure 43. S3 static 웹서버 설정 화면.**</center>

1. `Overview` 탭 선택 -> `Upload` 선택

1. 생성된 S3 Bucket에 이전 단계에서 생성된 파일들을 Drag & Drop으로
    업로드 합니다.

1. 이때 `Set permissions`을 Figure 44와 같이 반드시 `Grant public read
    access to this object(s)`로 설정해야 합니다.

    ![](/images/apps/internet_facing_app/image55.png?width="4.247341426071741in"
    height="3.4674245406824147in)

    <center>**Figure 44. S3 파일들에 대한 Make public 설정 화면.**</center>

### 최종 서비스 테스트하기

1.  웹브라우즈를 구동하시고 S3 Endpoint URL에 접속합니다 (Figure 45
    참조)

1.  Translate to German 오른편의 텍스트 입력 창에 영문 문장을
    입력합니다. (Ex. "I love you")

1.  몇 초 정도 기다리시면 하단에 번역 결과가 보여집니다.

![](/images/apps/internet_facing_app/image56.png?width="6.988888888888889in"
height="2.779166666666667in)

<center>**Figure 45. 웹기반 번역 서비스 테스트 화면.**</center>

### SageMaker Endpoint 서버 자동 확장 설정하기

본 섹션은 향후 실제 필요시에 대한 참조용으로 제공됩니다. 실제 Hands-on을
하실 필요는 없습니다.

웹 기반 서비스를 제공하기 시작하고 사용자 수가 증가하기 시작하면
SageMaker의 Inference 서버도 자동으로 확장되게 설정하실 수 있습니다.

![](/images/apps/internet_facing_app/image57.png?width="6.988888888888889in"
height="2.390277777777778in)

<center>**Figure 46. Endpoint 설정에서 InitialInstanceCount 변수 화면.**</center>

Figure 46와 같이 Endpoint 서버 설정에서의 Instance count는
`InitialInstanceCount`로 설정이 됩니다. 즉 초기의 서버 갯수 만을
설정하는 것이고 사용자의 요청 부하에 따라 서버 설정이 바뀌게 할 수
있습니다. 아래에는 AWS SageMaker 콘솔을 이용해서 `autoscaling` 을 설정
하는 방법을 보겠습니다.

1.  AWS SageMaker 콘솔에서 왼편의 `Endpoints`를 선택하신 후 오른편
    화면에서 생성하신 Endpoint를 선택합니다 (Figure 47 참조).

    ![](/images/apps/internet_facing_app/image58.png?width="6.988888888888889in"
    height="2.459722222222222in)

    <center>**Figure 47. AWS 콘솔에서 SageMaker의 Endpoints 선택 화면**</center>

1.  선택된 Endpoint 내용 화면에서 스크롤을 하셔서 **Endpoint runtime
    settings** 에서 `AllTraffic`을 선택하신 후 오른편의 **Configure
    auto scaling** 버튼을 선택합니다. 참고로 이 화면에서 각 Variant별
    Weight 변경 (`Update Weigths`)와 평상시 서버 개수 (**Update
    Instance count**) 도 변경하실 수 있습니다.

    ![](/images/apps/internet_facing_app/image59.png?width="6.988888888888889in"
    height="1.2041666666666666in)

    <center>**Figure 48. Auto scaling 설정 화면.**</center>

1.  Configure variant automatic scaling 화면에서는 Variant automatic
    scaling과 Scaling policy를 설정 하실 수 있습니다
    ([참조링크](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling.html)).
    Amazon SageMaker는 [target-tracking scaling
    정책](https://docs.aws.amazon.com/autoscaling/application/userguide/application-auto-scaling-target-tracking.html)을
    사용하고 있습니다. 즉 미리 정의된 metric이나 custom metric을
    사용하셔서 target value를 지정하실 수 있는데 CloudWatch 알람을 통해
    scaling 정책을 구동 시키고 instance server scale을 조정하실 수
    있습니다. 본 핸즈온에서는 직접 다루지는 않지만
    [참조링크](https://docs.aws.amazon.com/autoscaling/application/userguide/application-auto-scaling-target-tracking.html)를
    통해 좀 더 자세한 내용을 파악해 보시는 것도 좋을 것 같습니다.

    ![](/images/apps/internet_facing_app/image60.png?width="4.9524606299212595in"
    height="6.701863517060367in)

    <center>**Figure 49. Automatic scaling 정책 설정 화면.**</center>

이상으로 본 모듈의 실습 과정을 마무리 하셨습니다. 워크 샵 이후 발생되는
비용을 방지하기 위해 다음 페이지의 서비스 종료 가이드를 통해 사용하신
리소스들을 모두 종료/삭제 해주십시오.