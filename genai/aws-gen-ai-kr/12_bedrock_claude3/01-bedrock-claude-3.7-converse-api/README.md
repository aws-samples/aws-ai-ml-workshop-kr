# Claude 3.7 Sonnet on Amazon Bedrock

## 1. Amazon Bedrock Converse API 사용 실습
- [01-Bedrock-Converse-API.ipynb](01-Bedrock-Converse-API.ipynb)


## 2. Reasoning 유무에 따른 답변 비교
### 2.1 프롬프트
- 신선식품의 폐기율을 5% 줄이려면 어떤 전략을 세워야 할까요?

### 2.2 결과
- 프롬프트에 따라 다르겠지만, 제공된 프롬프트는 Reasoning 을 통한 답변이 더 좋은 결과를 나타냄. 
- Reasoning Generation Response 의 답변은 이후에 Reasioning 내용이 있습니다.

### 2.3 답변 내용
---

| Standard Generation Response | Reasoning Generation Response |
|------------------------------|-------------------------------|
| **표준 생성 응답에서는 모델이 바로 답변을 생성합니다.** | **추론 생성 응답에서는 모델이 먼저 내부적으로 추론 과정확한 답변을 생성합니다.** |
| **# 재고 관리 최적화** | **# 재고 관리 최적화** |
| FIFO(First In & First Out) 방법을 엄격히 적용할 것을 권장합니다. | 신선식품의 높은 폐기율의 문제에 수식 해결책: 재고 관리 최적화, 보관 솔루션, 소매 전략, 소비자 교육, 대안 활용, 기술 솔루션 등으로 접근할 수 있습니다. |
| FIFO(First In First Out) 원칙 엄격히 적용 | **# 공급망 최적화** |
| 유통기한 접근 상품부터 시선 노출 | • 개선된 수요 예측(데이터 분석 활용) |
| **# 유통기한 관리** | • 재고 회전율 향상 |
| • 유통기한 임박 상품 일괄적으로 드러내 진열 | • 운송 중 시간 단축 |
| • 색 코드를 활용하여 유통기간을 식별 쉽게 시각화 | **# 보관 솔루션** |
| • 손상되거 오래된 재품 격리 | • 신선도 유지를 위한 온도 조절 강화 |
| **# 소매 전략 개선** | • 수정된 컴플라이언스 포장 |
| • 유통기간 신속 확 확인 | • 과일 및 채소를 위한 개별 포장 |
| • 근접 유통기간 상품에 대해 '가격 조정' | • 직원 교육 강화(적절한 보관 방법) |
| • 벌크에도 포함될 할인 프로그램 신선도, 신뢰도 유지 | **# 소매 전략** |
| **# 재고 추적** | • 접근 만료일에 대한 가격인하 |
| • 재입고 수준을 필요한 양만 연결 | • 스마트 재고 표시(더 자주 업데이트) |
| • 고객요구에 맞춘 재고 확보 | • 판매 문제 기록 |
| **# 공급업체 개선** | **# 소비자 교육** |
| • 명확한 유통기간 약정을 맺어야 | • 유효기간 대 품질일자 정보 이해 |
| • 밀착 후기 공유 그것을 품질 보증 | • 적절한 저장 방법에 대한 명확한 안내 제공 |
| | • 음식물 쓰레기 인식 캠페인 |
| 이러한 솔루션을 적용한 회사는 신선식품의 재고관리를 효과적으로 처리하는데 큰 성공을 거두었습니다. | **# 대안 활용** |
| | • 판매불가 신선식품을 준비된 음식으로 전환 |
| | • 잉여작물 사용하여 사료나 퇴비로 전환 |
| | • 바이오가스 생산을 위한 폐기물 활용 |
| | **# 기술적 솔루션** |
| | • 신선도 표시 스마트 패키지 |
| | • 운송 및 보관 기간 동안 IoT 센서 감시 |
| | • 재고관리에 블록체인 솔루션 적용 |
| | |
| | 이러한 전략들을 통합적으로 실행하면 신선식품의 재고관리 문제를 크게 줄일 수 있습니다. |
| | **# reasoning** |
| | 신선식품의 폐기율이 높은 문제를 해결하기 위해선 다음과 같은 접근 방식을 고려할 수 있습니다: |
| | 1. Supply Chain Optimization: |
| | - Improved demand forecasting using data analytics |
| | - Better inventory rotation |
| | - Shorter supply chains to reduce transit time |
| | - Optimizing temperature control during transportation |
| | 2. Storage Solutions: |
| | - Enhanced temperature and humidity controls |
| | - Modified atmosphere packaging |
| | - Intuitive dividers for fruits and vegetables |
| | - Better training for staff on proper storage methods |
| | 3. Retail Strategies: |
| | - Dynamic pricing for products approaching expiration date |
| | - Smaller display quantities with more frequent restocking |
| | - Clear rotation of stock (FIFO - First In, First Out) |
| | - Optimized order quantities based on sales patterns |
| | 4. Consumer Education: |
| | - Clear information about storage methods |
| | - Better understanding of expiration dates vs. quality dates |
| | - Awareness campaigns about food waste |
| | 5. Alternative Uses: |
| | - Processing slightly imperfect items into prepared foods |
| | - Donating safe but unsalable items to food banks |
| | - Converting waste into compost or animal feed |
| | - Using for biogas production |
| | 6. Technology Solutions: |
| | - Smart packages that indicate actual freshness |
| | - Blockchain for traceability |
| | - IoT sensors for monitoring storage conditions |
| | - Advanced inventory systems and metrics |
| | |
| | I'll structure my answer to cover these main categories while providing specific actionable recommendations. |