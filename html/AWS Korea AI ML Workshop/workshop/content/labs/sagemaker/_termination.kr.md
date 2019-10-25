+++
title = "서비스 종료 가이드"
menuTitle = "서비스 종료 가이드"
date = 2019-10-15T15:18:03+09:00
weight = 299
+++
{{% notice tip %}}
AWS Event 엔진을 통해 임시로 생성하신 AWS Account의 경우 자동 삭제 되므로 아래의 작업이 필요없습니다.
{{% /notice %}}

{{% notice warning %}}
워크 샵 이후 발생 되는 비용을 방지하기 위해서 아래의 단계에 따라 모두 삭제하십시오.
{{% /notice %}}

### Notebook instance

먼저 Notebook instance를 stop 시킨 후에 삭제할 수 있습니다. 먼저 Actions 버튼을 누르시고 아래에 있는 stop을 선택합니다. 

![stop_instance](/images/termination/stop_instance.png?classes=border)

일단stop 이 되면 Actions 버튼에 Delete 메뉴가 활성화되어 선택할 수 있습니다. 

![stop_instance_1](/images/termination/stop_instance_1.png?classes=border)

만약 향후 사용을 위해 인스턴스를 삭제하지 않는다면, 스토리지 비용이 발생합니다. 중지된 인스턴스를다시 시작하려면 Start를 선택하면 됩니다.

### Endpoint

노트북에서 삭제하지 않은 Endpoint가 있다면 콘솔에서 수동으로 삭제할 수 있습니다. 좌측 메뉴에서 Endpoints를 선택하면, Endpoint들의 목록이 나오는데 이 중에서 초록색 InService 로 표시된 것들이 현재 가동중인 Endpoint 인스턴스들입니다. 삭제 방법은 Notebook instance의 삭제 방법과 동일합니다. 
{{% notice warning %}}
Endpoint의 삭제는 잊기 쉬우므로 특별히 주의합니다.
{{% /notice %}}

![endpoints](/images/termination/endpoints.png?classes=border)

### Amazon S3 bucket
실습 중에 생성된 S3 Bucket들을 모두 삭제합니다. 처음보는 낯선 이름일 수도 있으므로 주의합니다.

![s3_bucket_list](/images/termination/s3_bucket_list.png?classes=border)

이상으로 본 핸즈온 세션의 모든 과정을 마무리 하셨습니다. 수고하셨습니다.