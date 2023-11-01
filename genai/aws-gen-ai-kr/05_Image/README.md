# Lab 5 - 이미지 생성

## 1. 개요


이미지 생성은 이미지의 도움으로 자신의 생각을 표현하는 아티스트, 디자이너 및 콘텐츠 제작자에게 지루한 작업이 될 수 있습니다. FM(Foundation Models) 의 도움으로 이 깔끔한 작업은 예술가의 생각을 표현하는 단 한 줄의 텍스트로 간소화될 수 있으며, FM은 언어 프롬프트를 사용하여 다양한 주제, 환경 및 장면에 대한 사실적이고 예술적인 이미지를 만드는 데 사용될 수 있습니다. 

이 실습에서는 Amazon Bedrock에서 사용할 수 있는 기반 모델을 사용하여 이미지를 생성하고 기존 이미지를 수정하는 방법을 살펴보겠습니다.


## 2. 이미지 프롬프팅 (Image prompting)

좋은 프롬프트를 작성하는 것은 때로는 예술이 될 수 있습니다. 특정 프롬프트가 주어진 모델에 대해 만족스러운 이미지를 생성할지 여부를 예측하는 것은 종종 어렵습니다. 그러나 작동하는 것으로 관찰된 특정 템플릿이 있습니다. 
대체로 프롬프트는 
- (i) 이미지 유형 (photograph/sketch/painting etc.)
- (ii) 설명 (subject/object/environment/scene etc.) 
- (iii) 스타일의  (realistic/artistic/type of art etc.)

세 부분으로 크게 나눌 수 있습니다 . 세 부분을 각각 개별적으로 변경하여 이미지 변형을 생성할 수 있습니다. <font color="red">형용사</font>는 이미지 생성 과정에서 중요한 역할을 하는 것으로 알려져 있습니다. 또한 더 많은 세부 정보를 추가하면 생성 과정에 도움이 됩니다. 

사실적인 이미지를 생성하려면  
- “a photo of”, “a photograph of”, “realistic” or “hyper realistic” 과 같은 문구를 사용할 수 있습니다. 

아티스트의 이미지를 생성하려면 
- “by Pablo Piccaso” or “oil painting by Rembrandt” or “landscape art by Frederic Edwin Church” or “pencil drawing by Albrecht Dürer”과 같은 문구를 사용할 수 있습니다. 

다양한 아티스트를 결합할 수도 있습니다. 카테고리별로 예술적인 이미지를 생성하려면 
- “lion on a beach, abstract”. Some other categories include “oil painting”, “pencil drawing, “pop art”, “digital art”, “anime”, “cartoon”, “futurism”, “watercolor”, “manga” 등이 포함됩니다. 

또한 다음과 같은 세부정보를 포함할 수도 있습니다. 
- lighting or camera lens such as 35mm wide lens or 85mm wide lens and details about the framing (portrait/landscape/close up etc.).


동일한 프롬프트가 여러 번 제공되더라도 모델은 다른 이미지를 생성합니다. 따라서 여러 이미지를 생성하고 애플리케이션에 가장 적합한 이미지를 선택할 수 있습니다.

## 3. 파운데이션 모델 (Foundation Model)

이 기능을 제공하기 위해 Amazon Bedrock은 Stability AI의 이미지 생성을 위한 독점 기반 모델인 [Stable Diffusion XL](https://stability.ai/stablediffusion)을 지원합니다. Stable Diffusion은 확산 원리에 따라 작동하며 각각 다른 목적을 가진 여러 모델로 구성됩니다.

Stable Diffusion의 이미지는 아래 3가지 주요 모델에 의해 생성됩니다.
1. CLIP 텍스트 인코더 (입력 텍스트의 토큰 임베딩 변환)
2. UNet (Diffusion 스텝)
    - 토큰 임베딩과 노이즈 이미지를 입력 받아서, 노이즈 제거 과정을 통해 이미지 텐서 생성
3. VAE 디코더 (이미지 텐서로 실제 이미지 생성)



작동 방식은 다음 아키텍처로 설명할 수 있습니다:
![Stable Diffusion Architecture](./images/sd.png)


## 4. 패턴

이 워크숍에서는 Amazon Bedrock을 사용하여 이미지 생성에 대한 다음 패턴을 배울 수 있습니다.

1. Text to Image
    ![Text to Image](./images/71-txt-2-img.png)
2. Image to Image (In-paiting)
    ![Text to Image](./images/72-img-2-img.png)

## 5. 관련 노트북
- 01_UX_Bedrock Stable Diffusion XL.ipynb
    - Amazon Bedrock Image Playground 에서 프롬프트를 넣고 실습을 합니다.
- 02_Bedrock Stable Diffusion XL.ipynb
    - Amazon Bedrock API 로 Text-To-Image, Image-To-Image 를 실습 합니다.

