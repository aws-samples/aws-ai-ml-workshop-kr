# 개발된 LEX 봇을 다른 곳에서 사용하기 위한 Cloud Formation 생성 방법
---

# 1. 배경
- 이 문서는 개발이 된 Lex 봇을 다른 곳으로 마이그레이션을 하기 위해서 CloudFormation (CF) 파일을 만드는 방법을 소개 합니다. 

# 2. 폴더 구조 및 파일 요약
- 먼저 간단하게 주어진 깃 피라지토리의 전체 폴더 및 주요 파일을 소개 합니다.

```
 |-README.md ----------------------------------: 리드미 파일
 |-1.1.Prepare-Artifact_EventEngine_S3.ipynb --: 패키징 및 에벤트 엔진 S3 버킷으로 업로딩
 |-1.2.Prepare-Artifact_Public_S3.ipynb -------: 패키징 및 개인 S3 버킷으로 업로딩
 |-9.1.Run-CF_Test.ipynb ----------------------: CF CMD 로 테스트
 |-Artifact -----------------------------------: 패키징을 할 소스 코드 및 아티펙트
 | |-Lambda-BizLogic-DynamoDB-Code ------------: DDB 관련 코드
 | | |-lambda_function.py ---------------------: 레코드 입력 함수
 | | |-cfnresponse.py -------------------------: 작업 완료 및 에러시에 CF 에게 시그널 보내는 함수 목록
 | | |-BankingBotData.json --------------------: 테이블에 입력될 레코드
 | |-Lambda-LexArtifact-Export-Artifact -------: 렉스 임포트를 위해 필요한 코드 및 아티펙트
 | | |-lambda_function.py  --------------------: 렉스를 임포트하기 위한 람다 함수
 | | |-BankingBot-DRAFT-QWXG0RALBK-LexJson.zip : 개발된 렉스를 익스포트하면 생기는 아티펙트 들
 | | |-bot_details.json -----------------------: Bot 명세 파일
 | | |-cfnresponse.py -------------------------: 작업 완료 및 에러시에 CF 에게 시그널 보내는 함수 목록
 | |-Lambda-BizLogic-BankingService-Code ------: 렉스 Biz 로직을 구현한 코드
 | | |-utils.js 
 | | |-banking_service.js
 | | |-fallback.js
 | | |-check_balance.js
 | | |-index.js
 | | |-transfer.js
 |-Deploy-Package -----------------------------: CF_KRBankingBot.yaml 를 제외하고, 위의 1.1 혹은 1.2 를 실행한 후에 생성이 된 배포 아티펙트
 | |-CF_KRBankingBot.yaml ---------------------: CF 용 YAML파일
 | |-Lambda-BizLogic-DynamoDB -----------------: DynamoDB 배포 폴더
 | | |-DDBInsert.zip --------------------------: DynamoDB 배포 아티펙트 압축 파일
 | |-Lambda-BizLogic-BankingService -----------: 렉스 비즈 로직 배포 폴더   
 | | |-BankingServiceFunction.zip -------------: 렉스 비즈 로직 배포 아티펙트 압축 파일
 | |-Lambda-LexArtifact-Import ----------------: 렉스 익스포트 아티펙트 폴더 
 | | |-LexImport.zip --------------------------: 렉스 익스포트 아티펙트 압축 파일
```

# 3. CF 생성을 위한 코드 준비
## 3.1 Lambda-BizLogic-DynamoDB-Code : DDB 관련 코드
- BankingBotData.json 에는 테이블의 입력 데이터가 있음. 테이블의 레코드가 수정이 되면, 이 파일을 업데이트 해야 합니다.
- lambda_function 는 주어진 테이블 이름에 BankingBotData.json 의 데이타를 인서트 하는 일을 함.

## 3.2 Lambda-LexArtifact-Export-Artifact : 렉스 임포트를 위해 필요한 코드 및 아티펙트
- "BankingBot-DRAFT-QWXG0RALBK-LexJson.zip" 는 렉스 익스포트를 하면 생성되는 zip 파일 입니다.
- "bot_details.json" 의 렉스의 익스포트한 zip 파일 이름을 명시 합니다.
    - bot_definition: "BankingBot-DRAFT-QWXG0RALBK-LexJson.zip"
- lambda_function 는 렉스 임포트를 수행하는 람다 함수 입니다.

## 3.3 Lambda-BizLogic-BankingService-Code 
- 렉스 Biz 로직을 구현한 코드


# 4. 배포 패키징 파일 준비
두개의 옵션이 있습니다. 두개의 차이점은 S3 버킷을 이벤트 엔진의 버킷을 사용을 할지 혹은 사용자 정의의 버킷을 사용할지 입니다. 이외는 모든 것이 같습니다.
- 1.1.Prepare-Artifact_EventEngine_S3.ipynb
    - 필요한 파일을 패키징을 하고 주어진 에벤트 엔진 S3 버킷으로 업로딩
        - [중요] 현재 코드에 있는 Key 정보는 세션 단위로 업데이트 됩니다. 세션을 열고 새로이 주어진 정보로 업데이트 해야 합니다.
- 1.2.Prepare-Artifact_Public_S3.ipynb -------: 패키징 및 개인 S3 버킷으로 업로딩
    - 퍼블릭으로 오픈이 된 버킷 이름을 명시 해야 합니다.


# 5. YAML 구조

## 5.1 아티펙트 Zip 코드
    LexImportSource:
      Name: 'LexImportCode/LexImport.zip'
    BusinessLogicSource:
      Name: 'Lambda-BankingService/BankingServiceFunction.zip'
    DBImportSource:
      Name: 'DDBCode/DDBInsert.zip'

## 5.2 파리미터 변수 이름 정의

```
Parameters:
  BankingServicesName: # 렉스 봇의 이름
  BankingLambdaFunctionName: # 렉스 봇의 비즈 로직이 있는 함수
  DynamoDBTableName: # 렉스 봇의 비즈 로직이 사용하는 DDB 테이블 이름
```

## 5.3 리소스 생성

### 역할

- LexRole
    - Lex 봇 설치시에 할당되는 역할
- LambdaRole
    - 람다 함수를 실행할때의 역할
- LexImportRole    
    - Lex 임포트시의 사용되는 역할

### 주요 함수

- 렉스 봇을 설치하기 위한 사용자 정의 메인 함수
    - InvokeLexImportBankingServicesFunction:


- 렉스 봇의 모든 아테펙트 (봇 엘리아스, 인텐트 등) 설치하기 위한 람다 함수
    - LexImportBankingServicesFunction:


- 렉스 봇의 비즈니스 로직을 처리하기 위한 람다 함수
    - BankingServicesBusinessLogic:


- 렉스 못의 비즈니스 로직 처리 위한 람다 함수 실행 권한 추가
    - BankingServicesLambdaPermission:


- 다이나모 DB 테이블 정의
    - DynamoDBTable:


- 다이나모 DB 테이블 생성 요청을 위한 사용자 정의 함수
    - DynamoTableInsert:


- 다이나모 DB 테이블 레코드 입력을 위한 람다 함수
    - DynamoDBImport:

---
# A. 참고 리소스

- Build conversational experiences for credit card services using Amazon Lex
    - https://aws.amazon.com/ko/blogs/machine-learning/build-conversational-experiences-for-credit-card-services-using-amazon-lex/


- 봇 익스포트
    - https://docs.aws.amazon.com/ko_kr/lexv2/latest/dg/export.html


* 람다 함수 마이그레이션
    * https://aws.amazon.com/ko/premiumsupport/knowledge-center/lambda-function-migration-console/


* Bulk data ingestion from S3 into DynamoDB via AWS Lambda
    * https://medium.com/analytics-vidhya/bulk-data-ingestion-from-s3-into-dynamodb-via-aws-lambda- (https://medium.com/analytics-vidhya/bulk-data-ingestion-from-s3-into-dynamodb-via-aws-lambda-b5bdc30bd5cd)

