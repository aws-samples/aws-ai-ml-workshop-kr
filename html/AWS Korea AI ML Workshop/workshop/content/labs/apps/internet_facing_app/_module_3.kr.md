+++
title = "Module 3: 영어-독어 번역 ML 모델 학습"
menuTitle = "영어-독어 번역 ML 모델 학습"
date = 2019-10-23T15:50:20+09:00
weight = 314
+++

#### Sequence-to-Sequence 알고리즘 노트북 열기

SageMaker가 지원하는 Seq2Seq 알고리즘은 MXNet 기반으로 개발된
[Sockeye](https://github.com/awslabs/sockeye) 알고리즘을 기반으로 개발된
최신의 Encoder-decoder 구조를 구현한 것으로 문서자동요약이나 언어 번역
서비스에 적용할 수 있습니다.

실습을 위해서 현재 설치되어 있는 SageMaker의 Jupyter 노트북의 예제들 중
아래의 디렉토리에 위한 Jupyter 노트북을 실행하시면 됩니다 (Figure 5
참조).

    /aws-ai-ml-workshop-kr/src/release/2018-11/module8-SageMaker-Seq2Seq-Translation-English-German-InternetFacingApp.ipynb

![](/images/apps/internet_facing_app/image15.png?width="6.988888888888889in"
height="1.6222222222222222in)

<center>**Figure 5. Seq2Seq 노트북 디렉토리 위치.**</center>

![](/images/apps/internet_facing_app/image16.png?width="6.65803258967629in"
height="3.2244892825896763in)

<center>**Figure 6. 노트북 화면.**</center>

### 노트북에 대한 설명

본 노트북은 아래에 위치한 예제 노트북의 수정된 버전으로 미리 학습된 머신
러닝 모델을 사용하도록 바뀌었습니다.

    /sample-notebooks/introduction\_to\_amazon\_algorithms/seq2seq\_translation\_en-de/SageMaker-Seq2Seq-Translation-English-German.ipynb

상기 노트북은 빠른 학습 시간을 위해 Figure 7와 같이 전체 데이터 중
첫번째 10000개의 데이터의 대해서만 학습을 해서 Seq2Seq 알고리즘의
사용방법을 소개하고 있습니다.

![](/images/apps/internet_facing_app/image17.png?width="6.988888888888889in"
height="1.1583333333333334in)

<center>**Figure 7. 샘플 데이터 선택 화면.**</center>

Figure 8는 다운받은 corpus의 실제 데이터 내용으로 영어 및 독일어
데이터가 어떻게 문장 대 문장으로 매핑 되고 있는지를 보여주고 있습니다.

![](/images/apps/internet_facing_app/image18.png?width="3.4231364829396327in" height="2.394903762029746in)   
<center>**영문 데이터 (corpuc.tc.en.small 내용)**</center>
  
![](/images/apps/internet_facing_app/image19.png?width="3.1783431758530183in" height="2.403966535433071in)
<center>**독일어 데이터(corpuc.tc.de.small 내용)**</center>
                                                        
<center>**Figure 8. 번역기 학습을 위한 영문 자료와 독일어 자료 비교 화면.**</center>

실제로는10000개의 샘플 문장으로 훈련한 번역기는 좋은 결과를 보여줄 수
없습니다. 그렇지만 전체 데이터 학습을 위해서는 선택하시는 SageMaker의
서버 Instance Type에 따라 다르지만 수시간에서 수일의 장시간이 소요될 수
있습니다. 따라서 이 노트북의 개발자들은 좀더 나은 품질의 번역 결과
체험을 원하시는 사용자들 위해 전체 데이터에 이미 훈련이 된 모델을
공유하고 있습니다.

이 Pre-trained model을 사용하기 위해서는 노트북의 코드 중 Endpoint
Configuration 직전의 코드를 아래와 같이 수정해서 이미 훈련된 모델을
다운로드 한 다음 본인의 S3 버켓으로 업로드 하시면 됩니다. 이때 Jupyter
노트북 마지막 줄의 `sage.delete_endpoint` 는 데모를 계속 진행하기 위해
실행하지 않습니다. 이를 위해 이번에는 가장 마지막 줄에 있는 코드를 주석
처리하겠습니다.

![](/images/apps/internet_facing_app/image20.png?width="6.988888888888889in"
height="0.8027777777777778in)

<center>**Figure 9. `delete_endpoint` 함수 콜 코멘트 처리 화면.**</center>

### Pre-trained 모델을 사용 하기 위한 노트북 수정

노트북에서 하단의 S3 bucket 이름에 상기 생성한 S3 이름을 입력하시고
우측의 예와 비슷한 형식으로 prefix를 입력하시면 됩니다 (Figure 11 참조).

![](/images/apps/internet_facing_app/image21.png?width="6.988888888888889in"
height="0.6555555555555556in)

<center>**Figure 10. 노트북 S3 버킷 이름 및 prefix 수정 전 화면.**</center>

![](/images/apps/internet_facing_app/image22.png?width="6.988888888888889in"
height="0.6506944444444445in)

<center>**Figure 11. S3 버킷 및 prefix 수정 후 화면 예제. 본인의 S3 버킷 이름으로 수정하셔야 합니다.**</center>

### 노트북 실행 방법

이제 노트북 전체를 실행할 준비가 되었습니다. Jupyter 노트북을 실행하는
방법은 코드가 있는 셀을 클릭으로 선택하신 후 Shift-enter 키를 누르시거나
또는 Jupyter 노트북 상단의 툴바에서 "Run cell, select below" 버튼을
클릭하셔도 됩니다.

![](/images/apps/internet_facing_app/image23.png?width="1.25in"
height="0.8611111111111112in)

<center>**Figure 12. Jupyter 노트북 셀 실행 툴바.**</center>

전체 실행 과정은 약 12분에서 15분 정도 소요 됩니다. 각각의 셀을
실행시키면서 셀 하단에 표시되는 처리결과들을 확인해 보시기 바랍니다.

노트북 코드 중 `Create endpoint configuration` 셀에서 현재
`InstanceType`이 `ml.m4.large` 로 되어 있습니다 (Figure 13 참조).
Seq2Seq 알고리즘은 Neural network 기반이기 때문에 `ml.p2.xlarge` (GPU)
instance를 사용하실 수 있지만 본 실습에서는 Free tier가 지원되는
`ml.m4.xlarge`* 를 사용하고 있습니다. `ml.t2.*` instance는 time-out 문제가
발생할 수 있으므로 본 실습에서는 사용하지 않습니다.

![](/images/apps/internet_facing_app/image24.png?width="6.393627515310587in"
height="2.142856517935258in)

<center>**Figure 13. Endpoint configuration 화면.**</center>

노트북 코드 중 `Create endpoint` 셀은 새로운 서버를 설치하고 실행
코드를 설치하는 과정이므로 본 노트북에서는 가장 많은 시간 (약
10\~11여분)이 소요 되는데 아래와 같은 메세지를 확인하시면 다음 모듈로
진행하시면 됩니다 (Figure 14참조).

![](/images/apps/internet_facing_app/image25.png?width="6.212043963254593in"
height="3.644886264216973in)

<center>**Figure 14. SageMaker Endpoint 생성 결과 화면.**</center>

노트북 가장 하단의 `delete_endpoint`는 주석 처리 되어 있어야 endpoint
서버가 다음 실습을 위해 계속 운용될 수 있습니다. 만약에 실행 전에
수정하셨다면 `Create endpoint` 부분의 코드를 다시 실행하시기
바랍니다.
