워크샵에 사용된 리소스 정리
====================================================

AWS에 인프라 배포는 비용이 발생됩니다. AWS 이벤트에 참석하는 경우 크레딧이 제공됩니다. 워크샵을 마쳤으면 아래를 통해 모든 것이 삭제되었는지 확인하여 불필요한 과금이 발생되지 않도록 합니다.

### 1. 개별 리소스 제거
- 아래 노트북을 실행하여 개별적인 리소스틀 제거 합니다.
    - [Cleanup.ipynb](../CleanUp.ipynb)
    
### 2. 역할, 권한 및 SageMaker Notebook 을 삭제
- Setup_Environment 의 아래 화면 처첨 생성된 Cloud Formation Stack 을 제거하여 생성되었던 역할, 권한 및 SageMaker Notebook 을 삭제 합니다.
- CloudFormation의 상단 삭제 버튼을 클릭해서 워크숍에서 사용된 [CloudFormation 스택을 삭제합니다](http://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/cfn-console-delete-stack.html). 
![IAM resources acknowledgement](images/cf-03.png)
스택 삭제 프로세스에 오류가 발생하면 CloudFormation 대시 보드에서 이벤트 탭을 보고 실패한 단계를 확인합니다. CloudFormation에서 관리하는 리소스에 연결된 수동으로 생성된 리소스를 정리해야하는 경우 일 수 있습니다.

