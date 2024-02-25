<h1 align="left"><b>EC2에 도메인 적용 및 ACM 인증서 발급</b></h1>
본 PoC 서비스에 Amazon Route53 통해 awssa.site 도메인을 연결하는 과정입니다.
<p align="center">
    <img src="./images/00_architecture.png"  width="900" height="470">
</p>

## <div id="Contents">**디렉토리 구조**</div>
Streamlit on EC2가 동작되고, 구동할 수 있는 상태를 확인합니다.

1. [도메인(awssa.site) 준비](./01_prepare_domain.md)
2. [Amazon Route53 도메인 인증](./02_setup_route53.md)
3. [AWS Certificate Manager인증서 발급 받기](./03_acm_certificate_manager.md)  
4. [Load Balancer에 Target Group 설정](./04_setup_load_balancer.md)
