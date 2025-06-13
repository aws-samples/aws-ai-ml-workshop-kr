# Amazon Bedrock 파운데이션 모델 비교 분석

KB금융그룹을 위한 모델 검토 자료

작성일: 2025년 06월 12일

## 목차

1. [요약](#요약)
2. [주요 발견사항](#주요-발견사항)
3. [모델 개요](#모델-개요)
4. [가격 분석](#가격-분석)
5. [기술 명세](#기술-명세)
6. [지역 가용성](#지역-가용성)
7. [엔터프라이즈 고려사항](#엔터프라이즈-고려사항)
8. [결론 및 권장 사항](#결론-및-권장-사항)
9. [참고 문헌](#참고-문헌)

## 요약

이 보고서는 KB금융그룹이 요청한 Amazon Bedrock 파운데이션 모델에 대한 종합적인 비교 분석을 제공합니다. [1][2][3] 분석 대상 모델은 AP-Northeast-2(서울) 리전의 Nova Micro, Nova Lite, Nova Pro, Claude 3.5 Sonnet, Claude 3.5 Sonnet v2, Claude 3 Haiku와 미국 및 EU 리전의 Claude 3.7 Sonnet, Claude 3.5 Haiku입니다. [5][11] 이 분석에서는 모델별 주요 특징, 가격(입력 및 출력 요금), 컨텍스트 길이, 다중 모달 지원 여부, 지역 가용성 및 RI 정책을 중점적으로 다루고 있습니다. [6][7][10]

분석 결과, Nova 제품군은 비용 효율성과 다양한 지역 가용성에서 강점을 보이는 반면, Claude 모델은 고급 기능과 정확도에서 우수한 성능을 제공하는 것으로 나타났습니다. [12] KB금융그룹의 기존 Azure 기반 GenAI 플랫폼을 감안할 때, 이 보고서는 향후 에이전트 확장을 위한 모델 선택 및 통합 고려사항에 대한 인사이트를 제공합니다. [16]

## 주요 발견사항

- Nova 제품군은 타 모델에 비해 최소 75% 저렴하여 비용 효율성이 매우 뛰어납니다. [2][6]
- Claude 3.5 Sonnet v2와 Claude 3.7 Sonnet은 컴퓨터 사용 기능(베타)을 제공하여 고급 기능 측면에서 우위를 점합니다. [3]
- 모든 Claude 모델은 일관된 200K 토큰의 컨텍스트 윈도우를 제공합니다. [8]
- Nova Lite와 Pro는 이미지, 비디오, 텍스트를 포함한 다중 모달 기능을 지원합니다. [9]
- 배치 처리를 통해 대부분의 모델에서 최대 50% 할인된 가격으로 처리가 가능합니다. [10]
- Claude 3.7 Sonnet은 2025년 3분기, Claude 3.5 Haiku는 2025년 2분기에 서울 리전에서 사용 가능할 것으로 예상됩니다. [11]
- 금융 기관을 위한 규제 준수 및 보안 기능이 포괄적으로 제공됩니다. [13][15]
- KB금융그룹의 Azure 기반 플랫폼과의 통합을 위한 하이브리드 클라우드 접근 방식이 가능합니다. [16]

## 모델 개요

Amazon Bedrock은 AWS에서 제공하는 완전 관리형 서비스로, 다양한 AI 파운데이션 모델에 대한 액세스를 제공합니다. [1] 이 분석에서는 Nova 제품군과 Claude 제품군의 8개 모델을 중점적으로 다룹니다.

### Nova 모델 제품군

Amazon Nova는 2024년 12월에 발표된 새로운 세대의 파운데이션 모델입니다. [1][2] 세 가지 주요 모델이 있으며, 모두 200개 이상의 언어와 다양한 모달리티를 지원합니다:

- **Nova Micro**: 텍스트 전용 모델로, 가장 낮은 지연 시간과 매우 낮은 비용을 제공합니다. [1][12]
- **Nova Lite**: 이미지, 비디오, 텍스트를 처리할 수 있는 저비용 다중 모달 모델입니다. [1][9]
- **Nova Pro**: 최적의 정확도/속도/비용 균형을 갖춘 고성능 다중 모달 모델입니다. 지연 시간 최적화 버전도 제공됩니다. [1][12]

### Claude 모델 제품군

Anthropic의 Claude 모델은 다양한 기능과 성능 수준을 제공합니다: [3][4]

- **Claude 3.7 Sonnet**: 확장된 사고 능력을 갖춘 고성능 모델입니다. 컴퓨터 사용 기능(베타)을 지원합니다. [3][4]
- **Claude 3.5 Sonnet**: 지능형 범용 모델로, v2 버전은 컴퓨터 사용 기능(베타)을 지원합니다. [3][4]
- **Claude 3.5 Haiku**: 2024년 11월에 출시된 속도에 최적화된 모델입니다. [3][12]
- **Claude 3 Haiku**: Claude 제품군에서 가장 빠른 응답 시간을 제공하는 모델입니다. [4][12]

![모델 기능 비교 매트릭스](./artifacts/report_images/feature_matrix.png)
*그림 1: Amazon Bedrock 파운데이션 모델 기능 비교 매트릭스 [8][9]*

## 가격 분석

아래 표는 각 모델의 입력 및 출력 요금을 1백만 토큰(1M) 기준으로 정리한 것입니다. [6][7]

| 모델명 | 입력 요금(1M토큰) | 출력 요금(1M토큰) | 배치 입력(1M토큰) | 배치 출력(1M토큰) |
|---|---|---|---|---|
| Nova Micro | $40 | $140 | $20 | $70 |
| Nova Lite | $60 | $240 | $30 | $120 |
| Nova Pro | $800 | $3,200 | $400 | $1,600 |
| Nova Pro (지연 최적화) | $1,000 | $4,000 | 지원 안 함 | 지원 안 함 |
| Claude 3.7 Sonnet | $3,000 | $15,000 | 지원 안 함 | 지원 안 함 |
| Claude 3.5 Sonnet | $3,000 | $15,000 | $1,500 | $7,500 |
| Claude 3.5 Haiku | $800 | $4,000 | $500 | $2,500 |
| Claude 3 Haiku | $250 | $1,250 | $125 | $625 |

가격 분석에서 다음과 같은 중요한 점이 발견되었습니다: [6][7][10]

- Nova 모델은 동급 모델에 비해 최소 75% 저렴합니다. [2]
- 배치 처리를 활용하면 대부분의 모델에서 약 50%의 비용 절감이 가능합니다. [10]
- Nova Micro는 가장 비용 효율적인 텍스트 전용 모델을 제공합니다. [6][12]
- Claude 3.7 Sonnet 및 Claude 3.5 Sonnet은 가장 높은 가격대를 보이지만, 고급 기능을 제공합니다. [7]

![가격 비교 분석](./artifacts/report_images/pricing_comparison.png)
*그림 2: Amazon Bedrock 파운데이션 모델 가격 비교 분석 [6][7]*

### RI(Reserved Instance) 정책

Amazon Bedrock은 다음과 같은 비용 최적화 옵션을 제공합니다: [10]

- 일부 모델 및 지역에서 Provisioned Throughput 옵션 제공 (6개월 약정) [10]
- 배치 처리를 통해 정규 가격 대비 최대 50% 할인 [10]
- 비용 최적화를 위한 프롬프트 캐싱 기능 지원 [10]

## 기술 명세

### 컨텍스트 윈도우 길이

컨텍스트 윈도우 길이는 모델이 한 번에 처리할 수 있는 토큰 수를 나타냅니다: [8]

- Claude 3.7 Sonnet: 200K 토큰 [8]
- Claude 3.5 Sonnet: 200K 토큰 [8]
- Claude 3.5 Haiku: 200K 토큰 [8]
- Claude 3 Haiku: 200K 토큰 [8]
- Nova 모델: 모델 유형에 따라 다양한 컨텍스트 길이 제공 [8]

### 멀티 모달 기능

모델별 멀티 모달 지원 현황은 다음과 같습니다: [9]

- Nova Micro: 텍스트 전용 [9]
- Nova Lite: 멀티 모달 지원 (이미지, 비디오, 텍스트) [9]
- Nova Pro: 고급 멀티 모달 기능 [9]
- Claude 3.7 Sonnet: 컴퓨터 사용 기능(베타)이 포함된 멀티 모달 [3][9]
- Claude 3.5 Sonnet v2: 컴퓨터 사용 기능(베타)이 포함된 멀티 모달 [3][9]
- Claude 3.5 Haiku: 속도에 최적화된 텍스트 전용 [9]
- Claude 3 Haiku: 텍스트 전용 [9]

### 성능 벤치마크

모델별 주요 성능 특성은 다음과 같습니다: [12]

- Nova Micro: 최저 지연 시간 및 비용에 최적화 [12]
- Nova Lite: 다중 모달 작업을 위한 균형 잡힌 성능 [12]
- Nova Pro: 최적의 속도/비용 균형으로 최고의 정확도 [12]
- Claude 3.7 Sonnet: 고급 추론 기능 [12]
- Claude 3.5 Sonnet: 범용 지능 [12]
- Claude 3.5 Haiku: 가장 빠른 응답 시간 [12]

![기술 역량 분석](./artifacts/report_images/technical_capabilities.png)
*그림 3: Amazon Bedrock 파운데이션 모델 기술 역량 분석 [12]*

## 지역 가용성

### AP-Northeast-2(서울) 리전 가용성

서울 리전에서의 모델 가용성은 다음과 같습니다: [5][11]

- 현재 사용 가능:
  - Nova 제품군(모든 모델) [5]
  - Claude 3.5 Sonnet [5]
  - Claude 3 Haiku [5]
- 예정된 출시:
  - Claude 3.7 Sonnet: 2025년 3분기 예상 [11]
  - Claude 3.5 Haiku: 현재 프리뷰 상태, 2025년 2분기에 일반 가용성 예상 [11]

### 미국 및 EU 리전 가용성

미국 및 EU 리전에서의 모델 가용성은 다음과 같습니다: [5]

- US-East/West 리전:
  - 모든 모델 현재 사용 가능 [5]
  - 모든 모델에 대한 완전한 기능 지원 [5]
  - 새로운 기능을 위한 주요 배포 지역 [5]
- EU-Central/North/West 리전:
  - 대부분의 모델이 완전한 지원으로 사용 가능 [5]
  - Claude 3.5 Haiku: 제한적 가용성 [5]
  - Nova 제품군: 완전한 가용성 [5]

![지역 가용성 비교](./artifacts/report_images/regional_availability.png)
*그림 4: Amazon Bedrock 파운데이션 모델 지역 가용성 비교 [5][11]*

## 엔터프라이즈 고려사항

### 보안 및 규정 준수

Amazon Bedrock은 기업, 특히 금융 기관을 위한 포괄적인 보안 및 규정 준수 기능을 제공합니다: [13][14][15]

- 보안 인증:
  - ISO, SOC, CSA STAR Level 2 인증 [13]
  - HIPAA 적격 및 GDPR 준수 [13]
  - AWS GovCloud에서 FedRAMP High 승인 [13]
- 금융 기관을 위한 특별 기능:
  - 자동화된 규제 준수 모니터링 [15]
  - 자금 세탁 방지(AML) 규칙 지원 [15]
  - 은행 비밀법(BSA) 준수 [15]
  - 데이터 프라이버시 제어 및 감사 기능 [15]

### 기업 통합 기능

Amazon Bedrock은 기존 엔터프라이즈 시스템과의 통합을 위한 다양한 기능을 제공합니다: [14]

- 규제 준수를 위한 Knowledge Bases 통합 [14]
- 멀티 에이전트 오케스트레이션 프레임워크 지원 [14]
- 세분화된 액세스 제어 및 가드레일 [14]
- 데이터 프라이버시를 위한 프라이빗 모델 복사본 [14]
- VPC 통합 기능 [14]

### AWS-Azure 통합 고려사항

KB금융그룹의 기존 Azure 기반 GenAI 플랫폼과의 통합에 관한 고려사항: [16]

- 하이브리드 클라우드 접근 방식에서의 장단점:
  - AWS Bedrock: AWS 네이티브 워크로드에 더 적합 [16]
  - Azure: Microsoft 에코시스템 통합에 더 강점 [16]
- 엔터프라이즈 BI 도구와의 통합 과제 [16]
- 크로스 플랫폼 보안 고려사항 [16]

### 금융 부문 맞춤화

금융 부문을 위한 맞춤형 기능: [17][18]

- 금융 사용 사례를 위한 모델 미세 조정:
  - 금융 거래 처리의 입증된 성공 사례 [17]
  - 금융 텍스트 분석에서 24.6% 성능 향상 [17]
  - 미세 조정 후 금융 컨텐츠에서 91.2% F1 스코어 달성 [17]
- 한국 관련 특화 사례:
  - SK텔레콤 사례 연구가 보여주는 한국어 최적화 성공 [18]
  - 한국 특화 사용 사례에 대한 상당한 성능 개선 [18]
  - 현지 규정 준수 및 규제 조정 [18]
  - 기존 한국 금융 시스템과의 통합 [18]

## 결론 및 권장 사항

이 보고서의 분석 결과, Amazon Bedrock의 파운데이션 모델은 다양한 성능과 비용 체계를 제공하여 KB금융그룹의 요구 사항에 맞게 선택할 수 있는 유연성을 제공합니다. [1][2][12]

### 모델 선택 가이드라인

- 비용 효율성이 우선 순위인 경우: Nova Micro 또는 Nova Lite가 가장 비용 효율적인 옵션을 제공합니다. [6][12]
- 고급 추론 기능이 필요한 경우: Claude 3.7 Sonnet 또는 Claude 3.5 Sonnet이 최고의 성능을 제공합니다. [12]
- 균형 잡힌 성능이 필요한 경우: Nova Pro 또는 Claude 3.5 Sonnet이 좋은 선택입니다. [12]
- 한국 리전 가용성이 필수적인 경우: 현재 Nova 제품군, Claude 3.5 Sonnet, Claude 3 Haiku가 사용 가능하며, Claude 3.7 Sonnet과 Claude 3.5 Haiku는 각각 2025년 3분기와 2분기에 출시될 예정입니다. [5][11]

### KB금융그룹을 위한 제언

기존 Azure 기반 GenAI 플랫폼과의 통합 및 확장을 고려할 때, 다음 사항을 권장합니다: [16]

- 하이브리드 접근 방식: Azure 기반 플랫폼의 강점과 Amazon Bedrock의 고급 모델을 결합하여 최상의 성능을 제공하는 하이브리드 아키텍처를 고려하세요. [16]
- 단계적 통합: 현재 서울 리전에서 사용 가능한 모델(Nova 제품군, Claude 3.5 Sonnet, Claude 3 Haiku)부터 시작하여 점진적으로 확장하는 방식을 추천합니다. [5]
- 비용 최적화: 배치 처리 및 RI 정책을 활용하여 총 소유 비용을 최적화하세요. [10]
- 금융 특화 맞춤화: 금융 부문 사용 사례에 맞게 모델을 미세 조정하여 성능을 향상시키는 것을 고려하세요. [17]
- 보안 및 규정 준수: 금융 기관에 중요한 보안 인증과 규정 준수 기능을 활용하세요. [13][15]

> **참고:** KB금융그룹의 특정 요구 사항과 사용 사례에 따라 이러한 권장 사항을 조정할 필요가 있습니다. 모델 성능은 지속적으로 개선되고 있으며, 새로운 모델이 정기적으로 출시됩니다. [11]

## 참고 문헌

[1]: [Announcing Amazon Nova foundation models](https://aws.amazon.com/about-aws/whats-new/2024/12/amazon-nova-foundation-models-bedrock)
[2]: [Introducing Amazon Nova](https://www.aboutamazon.com/news/aws/amazon-nova-artificial-intelligence-bedrock-aws)
[3]: [Tool use - Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-tool-use.html)
[4]: [Models overview - Anthropic API](https://docs.anthropic.com/en/docs/about-claude/models/overview)
[5]: [Supported foundation models in Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html)
[6]: [AWS Bedrock Pricing - Metal Toad](https://www.metaltoad.com/blog/aws-bedrock-pricing)
[7]: [Pricing - Anthropic](https://www.anthropic.com/pricing)
[8]: [Amazon Bedrock Foundation Models Guide](https://medium.com/@richardhightower/amazon-bedrock-foundation-models-a-complete-guide-for-genai-use-cases-75beadb608eb)
[9]: [Introducing Amazon Nova](https://www.aboutamazon.com/news/aws/amazon-nova-artificial-intelligence-bedrock-aws)
[10]: [Amazon Bedrock Pricing - AWS](https://aws.amazon.com/bedrock/pricing/)
[11]: [Anthropic Claude models - AWS Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html)
[12]: [Amazon Bedrock Pricing Explained - Caylent](https://caylent.com/blog/amazon-bedrock-pricing-explained)
[13]: [Amazon Bedrock Security and Privacy - AWS](https://aws.amazon.com/bedrock/security-compliance/)
[14]: [Protect sensitive data in RAG applications with Amazon Bedrock](https://aws.amazon.com/blogs/machine-learning/protect-sensitive-data-in-rag-applications-with-amazon-bedrock/)
[15]: [Automating Regulatory Compliance with Amazon Bedrock](https://aws.amazon.com/blogs/machine-learning/automating-regulatory-compliance-a-multi-agent-solution-using-amazon-bedrock-and-crewai/)
[16]: [Azure AI Foundry vs. Amazon Bedrock Comparison](https://www.qservicesit.com/azure-ai-foundry-vs-amazon-bedrock)
[17]: [Fine-tuning for Claude 3 Haiku in Amazon Bedrock](https://www.anthropic.com/news/fine-tune-claude-3-haiku-ga)
[18]: [SK Telecom improves telco-specific Q&A with Amazon Bedrock](https://aws.amazon.com/blogs/machine-learning/sk-telecom-improves-telco-specific-qa-by-fine-tuning-anthropics-claude-models-in-amazon-bedrock/)
