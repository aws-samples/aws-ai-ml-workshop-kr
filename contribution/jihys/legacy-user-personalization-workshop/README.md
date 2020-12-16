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

- Amazon Personalize에 제공하는 다양한 알고리즘을 통해 솔루션(사용자 추천 모델)을 생성해 봅니다.

  

##### 캠페인 생성 

- 생성된 모델을 활용하여 개인화 결과를 얻어 봅니다.
- Real-time서빙과 Batch recommendation 차이에 대해 알아보고 적합한 적용 방식에 대해 알아봅니다. 
- 사용자 기반 추천/아이템 기반 추천/Cold-start 아이템 추천의 결과를 확인해 봅니다.

##### 성능 분석 

- 추천 모델 성능 지표에 대한 메트릭에 대해 알아 봅니다.
- 사용자 기반 추천/아이템 기반 추천/Cold-start 아이템 추천의 결과를 확인해 봅니다.



## 순서

이 LAB의 모든 작업은 순서대로 진행 되어야 하며, 이 전 모듈이 완료 되어야 다음 모듈을 실행 할 수 있습니다.

###### Module 0. 작업 환경 구성하기 

아래 '작업 환경 구성하기' 가이드에 따라 환경구성을 해줍니다.

###### Module1 . Validating and importing user-item-interaction data 

이 모듈에 데이터 정제 작업 및 S3 bucket 업로드 과정, Personalize Schema 정의 및 Data import작업을 진행합니다. `01_Validating_and_Importing_User_Item_Interaction_Data.ipynb` 의 가이드를 따라 합니다.

###### Module2.  Creating and evaluating your first solutions

이 모듈에서는 Amazon Personalize에서 제공하는 알고리즘(Recipe)을 기반하여 고객 데이터를 가지고 학습한 뒤 모델(Solution)을 생성하도록 합니다.  또한 기본으로 제공하는 metric에 대해 간략히 알아보도록 합니다. 

`02_Creating_and_Evaluating_Solutions.ipynb`의 가이드를 따라 합니다. 

###### Module3. Deploying_Campaigns_getRecommendation

이 모듈에서는 만들어진 모델을 서비스를 위해 배포 하는 과정에 대해 알아보니다. `3.Deploying_Campaigns_getRecommendation.ipynb` 의 가이드를 따라 해 봅니다.

1. Deployment and capacity planning

2. How to interact with a deployed solution (various approaches)

3. Real-time interactions

4. Batch exporting

   

###### Module 4. Create Event Tracker and ineracting with personalize 

이번 모듈에서는 Event Tracker를 생서한 뒤 임의로 Interaction 정보를 Personalize에 보내 최근 추천 결과에 어떤 변화가 있는지 알아봅니다.  `4.Create_EventTracker_and_view_Interactions.ipynb` 의 가이드를 따라 합니다. 

###### Module 5. Evaluation with test dataset

이번 모듈에서는 모듈 1에서 따로 분리했던 테스트용 데이터 세트를 활용하여 각각의 솔루션들의 성능을 추가로 확인해 보도록 합니다. 모델은 신규 아이템이 추가 되거나 시간이 지날수로 추천 정확도가 떨어질 수 있습니다. 이 모듈에서 진행하는 내용은 배포된 모델의 성능에 문제가 없는지 확인해 보는데 사용 될 수도 있습니다. 

###### Module 6. Cleaning Up

워 크샾이 끝난 후에 모든 자원을 삭제 하기 위해서는 `06_Clean_Up_Resources.ipynb` 노트북 가이드를 따라 합니다. 여기서는 배포된 모든 Personalize자원을 삭제 하는 방법에 대해 알려줍니다.



## 작업 환경 구성하기 

CloudFormation Template 을 활용하여 작업 환경을 구성합니다.

1. 인터넷 브라우저(Chrome, Firefox 권장)를 하나 더 연뒤에 AWS Account로 로그인 합니다.
2. 브라우저에 새로운 Tab 을 생성한 뒤 아래 링크를 클릭하여 CloudFormation을 통해 환경을 구축합니다. 



[![Launch Stack](https://camo.githubusercontent.com/210bb3bfeebe0dd2b4db57ef83837273e1a51891/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f636c6f7564666f726d6174696f6e2d6578616d706c65732f636c6f7564666f726d6174696f6e2d6c61756e63682d737461636b2e706e67)](https://console.aws.amazon.com/cloudformation/home#/stacks/new?stackName=PersonalizePOC&templateURL=https://jihys-personalize-ap-northeast-2.s3.ap-northeast-2.amazonaws.com/PersonalizePOC_v1.yaml)

추가 궁금한 사항은 아래 스크린 샷의 가이드를 따라 합니다. 

### Cloud Formation Wizard

아래의 "Next" 버튼을 눌러서 CloudFormation 설정 작업을 시작 합니다.

[![StackWizard](https://github.com/jihys/startup-personalize-workshop/blob/master/static/imgs/img1.png)](https://github.com/jihys/startup-personalize-workshop/blob/master/static/imgs/img1.png)







다음과 같은 설정을 해 줍니다.  

1. 고유한 S3 버킷 이름을 지정합니다. S3 이름은 모든 리전에서 고유하게 설정되어야 합니다. 따라서 본인의 이름을 영어 소문자로 더해 줍니다. (예: personalizelab-jihye-seo)
2. Stack name을 원하는 대로 변경해 변경해 주세요. (예: PersonalizeLab)
3. 노트북 이름을 변경해 주세요. (Optional)

1. SageMaker Notebook instance에 EBS 볼륨을 더 할당 하고 싶으신 경우(데이터 셋이 클경우) 더 큰 사이즈로 변경하세요.기본은 10GB 입니다. 

모든 작업이 끝나면 `Next` 버튼을 클릭합니다.

[![StackWizard2](https://github.com/jihys/startup-personalize-workshop/blob/master/static/imgs/img2.png)](https://github.com/jihys/startup-personalize-workshop/blob/master/static/imgs/img2.png)

해당 페이지에서 밑에까지 스크롤 다운 한뒤에  `Next` 버튼을 클릭합니다. 해당 페이지에서는 모든 default설정은 POC를 하는데 충분합니다. 필요시에만 변경 하도록 합니다. 

[![StackWizard3](https://github.com/jihys/startup-personalize-workshop/blob/master/static/imgs/img3.png)](https://github.com/jihys/startup-personalize-workshop/blob/master/static/imgs/img3.png)

다시 아래까지 스크롤 한 뒤, CloudFormation Template이 IAM 자원을 생성할 수 있는 권한을 줄수 있도록 box를 체크합니다. 그리고  `Create Stack` 을 클릭합니다.

[![StackWizard4](https://github.com/jihys/startup-personalize-workshop/blob/master/static/imgs/img4.png)](https://github.com/jihys/startup-personalize-workshop/blob/master/static/imgs/img4.png)



몇 분뒤 CloudFormation은 새로운 자원을 생성하게 됩니다. 프로지버닝 단계에서는 다음과 같은 화면이 보일 것 입니다.

[![StackWizard5](https://github.com/jihys/startup-personalize-workshop/blob/master/static/imgs/img5.png)](https://github.com/jihys/startup-personalize-workshop/blob/master/static/imgs/img5.png)

모든 작업이 완성이 된 후에는 Status가 글씨로 아래와 같이 "CREATE_COMPLETE" 로 보이게 됩니다. 



[![StackWizard5](https://github.com/jihys/startup-personalize-workshop/blob/master/static/imgs/img6.png)](https://github.com/jihys/startup-personalize-workshop/blob/master/static/imgs/img6.png)

이제 AWS Management 콘솔 페이지에서  `Services` 클릭 후  `SageMaker`서비스를 조회한 하여 클릭한 후 SageMaker메뉴로 이동 합니다.

[![StackWizard5](https://github.com/jihys/startup-personalize-workshop/blob/master/static/imgs/img7.png](https://github.com/jihys/startup-personalize-workshop/blob/master/static/imgs/img7.png)

SageMaker콘솔에서 Notebook에서 본인이 방금 생성한 노트북을 찾아 클릭합니다.



[![StackWizard5](https://github.com/jihys/startup-personalize-workshop/blob/master/static/imgs/img8.png)](https://github.com/jihys/startup-personalize-workshop/blob/master/static/imgs/img8.png)

선택한 SageMaker notebook 에서  `Open JupyterLab` 을 클릭합니다. 

[![StackWizard5](https://github.com/jihys/startup-personalize-workshop/blob/master/static/imgs/img9.png)](https://github.com/jihys/startup-personalize-workshop/blob/master/static/imgs/img9.png)



이는 workshop을 위해 필요한 Jupyter Notebook 환경을 오픈합니다. Jupyter Notebook은 데이터 사이언스를 위한 IDE환경입니다. `startup-personalize-workshop` 이라는 폴더가 자동으로 오픈 될 것입니다. 만약 해당 폴더위치가 아니라면 브라우저의 화면의 왼편에 Folder icon 을 클릭하여 문서 가이드 대로 따라 합니다.  

## 





