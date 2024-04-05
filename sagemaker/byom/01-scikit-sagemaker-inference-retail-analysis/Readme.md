
# Bring Your Own Model 

## 소매점 매출 예측 Linear Regression

## 개요
이 프로젝트에서는 일반적인 ML 학습 방식인 Scikit-learn과 XGBoost를 사용하여 로컬에서 학습한 모델을 이용하여, SageMaker의 Endpoint API 추론 기능을 활용하여 모델을 배포합니다. 

목표는 서로 다른 지역에 위치한 45개 매장의 역사적 판매 데이터를 기반으로 각 매장의 다음 연도 부서 전체 매출을 예측하고, 휴일 주간 할인이 미치는 영향을 모델링하며, 비즈니스에 가장 큰 영향을 주는 요인을 파악하고 권장사항을 제공하는 것입니다.

## 데이터 출처
- [Kaggle: Retail Data Analytics](https://www.kaggle.com/manjeetsingh/retaildataset)

## 데이터 세트 설명
1. **stores_data_set.csv**: 45개 매장에 대한 익명화된 정보 (매장 유형 및 규모)
2. **sales_data_set.csv**:
    - Store: 매장 번호
    - Dept: 부서 번호
    - Date: 주
    - Weekly_Sales: 해당 매장의 해당 부서 매출
    - IsHoliday: 주가 특별 휴일 주인지 여부
3. **Features_data_set.csv**:
    - Store: 매장 번호
    - Date: 주 단위
    - Temperature: 지역의 평균 온도
    - Fuel_Price: 지역의 연료 가격
    - MarkDown1-5: 프로모션 할인(가격인하) 관련 익명화된 데이터 (2011년 11월 이후에만 사용 가능, 모든 매장에서 사용되지 않음)
    - CPI: 소비자 물가 지수
    - Unemployment: 실업률
    - IsHoliday: 해당 주가 특별 휴일 주인지 여부

## 모델 내용
이 프로젝트에서는 Scikit-learn과 XGBoost를 사용하여 Linear Regression 모델을 구축합니다. 모델은 역사적 판매 데이터와 다양한 피처 데이터를 활용하여 각 매장의 다음 연도 부서 전체 매출을 예측합니다. 또한, 휴일 주간 할인이 매출에 미치는 영향을 모델링하고, 비즈니스에 가장 큰 영향을 주는 요인을 파악하여 권장사항을 제공합니다.

## 사용 방법
1. 데이터 세트를 다운로드하고 준비합니다.
2. 모델 학습 및 평가 코드를 실행합니다.
3. SageMaker에 모델을 배포하고 Endpoint API를 통해 추론을 수행합니다.

## 기여
이 프로젝트에 대한 의견, 개선 사항 또는 기여를 환영합니다. 새로운 기능 요청 또는 버그 보고는 GitHub 이슈를 통해 제출해 주시기 바랍니다.