# Startup Amazon Personalize workshop Guide

Amazon Personalize는 추천 / 개인화 모델을 빠르고 효과적으로 구축하고 확장 할 수있는 기계 학습 서비스입니다. 해당 자료는 [GitHub Sample Notebooks](https://github.com/aws-samples/amazon-personalize-samples) 의 예제를 종합하여 구성 되었습니다. 서비스에 대한 소개와 자세한 가이드가 필요한 경우 아래의 문서를 참고하도록 합니다.

- 서비스 소개: [Product Page](https://aws.amazon.com/personalize/)
- Personalize기술 문서:  [Product Docs](https://docs.aws.amazon.com/personalize/latest/dg/what-is-personalize.html) 



## 목표

이번 워크샾을 통해 프로그래밍 방식으로 Amazon Personalize를 활용하여 개인화 추천 결과를 구축하는 방법에 대해 알아봅니다.  워크샾을 통해 다음과 같은 내용을 배우게 됩니다.

##### 데이터 준비하기

- Dataset을 Amazon Personalize에 매핑해 봅니다.
- 향후 Validation 및 데이터 분석을 위해서 데이터셋을 Training용과 Test용으로 분리 합니다. 

##### 솔루션 생성하기

- Amazon Personalize에 제공하는 알고리즘 중 최신 알고르즘인 User-Persoanlization 솔루션(사용자 추천 모델)을 생성해 봅니다.
- ㄷ

##### 캠페인 생성 

- 생성된 모델을 활용하여 개인화 결과를 얻어 봅니다.
- Event Tracker를 통해 새로운 사용 인터렉션 정보가 실시간 추천에 어떤 영향을 미치는 지 확인해 봅니다. 
- Exploration Weight 을 조정하여 콜드아이템 추천 비율을 확인해 봅니다.

##### 성능 분석 

- 추천 모델 성능 지표에 대한 메트릭에 대해 알아 봅니다.
- 테스트 셋을 가지고 성능을 확인해 봅니다.
- Cold-start 아이템 추천의 결과를 확인해 봅니다.




## 작업 환경 구성하기 

CloudFormation Template 을 활용하여 작업 환경을 구성합니다.

1. 인터넷 브라우저(Chrome, Firefox 권장)를 하나 더 연뒤에 AWS Account로 로그인 합니다.
2. 브라우저에 새로운 Tab 을 생성한 뒤 아래 링크를 클릭하여 CloudFormation을 통해 환경을 구축합니다. 



[![Launch Stack](https://camo.githubusercontent.com/210bb3bfeebe0dd2b4db57ef83837273e1a51891/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f636c6f7564666f726d6174696f6e2d6578616d706c65732f636c6f7564666f726d6174696f6e2d6c61756e63682d737461636b2e706e67)](https://console.aws.amazon.com/cloudformation/home#/stacks/new?stackName=PersonalizePOC&templateURL=https://jihys-personalize-ap-northeast-2.s3.ap-northeast-2.amazonaws.com/PersonalizePOC_v1.yaml)

추가 궁금한 사항은 아래 스크린 샷의 가이드를 따라 합니다. 

### Cloud Formation Wizard

아래의 "Next" 버튼을 눌러서 CloudFormation 설정 작업을 시작 합니다.

[![StackWizard](images/imgs/img1.png)




다음과 같은 설정을 해 줍니다.  

1. 고유한 S3 버킷 이름을 지정합니다. S3 이름은 모든 리전에서 고유하게 설정되어야 합니다. 따라서 본인의 이름을 영어 소문자로 더해 줍니다. (예: personalizelab-jihye-seo)
2. Stack name을 원하는 대로 변경해 변경해 주세요. (예: PersonalizeLab)
3. 노트북 이름을 변경해 주세요. (Optional)

1. SageMaker Notebook instance에 EBS 볼륨을 더 할당 하고 싶으신 경우(데이터 셋이 클경우) 더 큰 사이즈로 변경하세요.기본은 10GB 입니다. 

모든 작업이 끝나면 `Next` 버튼을 클릭합니다.

[![StackWizard2](images/imgs/img2.png)](images/imgs/img2.png)

해당 페이지에서 밑에까지 스크롤 다운 한뒤에  `Next` 버튼을 클릭합니다. 해당 페이지에서는 모든 default설정은 POC를 하는데 충분합니다. 필요시에만 변경 하도록 합니다. 

[![StackWizard3](images/imgs/img3.png)](images/imgs/img3.png)

다시 아래까지 스크롤 한 뒤, CloudFormation Template이 IAM 자원을 생성할 수 있는 권한을 줄수 있도록 box를 체크합니다. 그리고  `Create Stack` 을 클릭합니다.

[![StackWizard4](images/imgs/img4.png)](images/imgs/img4.png)



몇 분뒤 CloudFormation은 새로운 자원을 생성하게 됩니다. 프로지버닝 단계에서는 다음과 같은 화면이 보일 것 입니다.

[![StackWizard5](images/imgs/img5.png)](images/imgs/img5.png)

모든 작업이 완성이 된 후에는 Status가 글씨로 아래와 같이 "CREATE_COMPLETE" 로 보이게 됩니다. 



[![StackWizard6](images/imgs/img6.png)](images/static/imgs/img6.png)

이제 AWS Management 콘솔 페이지에서  `Services` 클릭 후  `SageMaker`서비스를 조회한 하여 클릭한 후 SageMaker메뉴로 이동 합니다.

[![StackWizard7](images/imgs/img7.png)](images/imgs/img7.png)

SageMaker콘솔에서 Notebook에서 본인이 방금 생성한 노트북을 찾아 클릭합니다.



[![StackWizard8](images/imgs/img8.png)](images/imgs/img8.png)

선택한 SageMaker notebook 에서  `Open JupyterLab` 을 클릭합니다. 

[![StackWizard9](images/imgs/img9.png)](images/imgs/img9.png)



이는 workshop을 위해 필요한 Jupyter Notebook 환경을 오픈합니다. Jupyter Notebook은 데이터 사이언스를 위한 IDE환경입니다. `startup-personalize-workshop` 이라는 폴더가 자동으로 오픈 될 것입니다. 만약 해당 폴더위치가 아니라면 브라우저의 화면의 왼편에 Folder icon 을 클릭하여 문서 가이드 대로 따라 합니다.  

## 





