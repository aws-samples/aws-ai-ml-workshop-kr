# TrOCR for Korean Language (PoC)

## Overview

TrOCR has not yet released a multilingual model including Korean, so we trained a Korean model for PoC purpose. Based on this model, it is recommended to collect more data to additionally train the 1st stage or perform fine-tuning as the 2nd stage.

## Collecting data

### Text data
We created training data by processing three types of datasets.
- News summarization dataset: https://huggingface.co/datasets/daekeun-ml/naver-news-summarization-ko
- Naver Movie Sentiment Classification: https://github.com/e9t/nsmc
- Chatbot dataset: https://github.com/songys/Chatbot_data
For efficient data collection, each sentence was separated by a sentence separator library (Kiwi Python wrapper; https://github.com/bab2min/kiwipiepy), and as a result, 637,401 samples were collected.

### Image Data

Image data was generated with TextRecognitionDataGenerator (https://github.com/Belval/TextRecognitionDataGenerator) introduced in the TrOCR paper.
Below is a code snippet for generating images.
```shell
python3 ./trdg/run.py -i ocr_dataset_poc.txt -w 5 -t {num_cores} -f 64 -l ko -c {num_samples} -na 2 --output_dir {dataset_dir}
```

## Training

### Base model
The encoder model used `facebook/deit-base-distilled-patch16-384` and the decoder model used `klue/roberta-base`. It is easier than training by starting weights from `microsoft/trocr-base-stage1`.

### Parameters
We used heuristic parameters without separate hyperparameter tuning.
- learning_rate = 4e-5
- epochs = 25
- fp16 = True
- max_length = 64

## Usage

### inference.py

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer
import requests 
from io import BytesIO
from PIL import Image

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten") 
model = VisionEncoderDecoderModel.from_pretrained("daekeun-ml/ko-trocr-base-nsmc-news-chatbot")
tokenizer = AutoTokenizer.from_pretrained("daekeun-ml/ko-trocr-base-nsmc-news-chatbot")

url = "https://raw.githubusercontent.com/aws-samples/aws-ai-ml-workshop-kr/master/sagemaker/sm-kornlp/trocr/sample_imgs/news_1.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

pixel_values = processor(img, return_tensors="pt").pixel_values 
generated_ids = model.generate(pixel_values, max_length=64)
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] 
print(generated_text)
```

All the code required for data collection and model training has been published on the author's Github.
- https://github.com/daekeun-ml/sm-kornlp-usecases/tree/main/trocr

### inference.py

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer
import requests 
from io import BytesIO
from PIL import Image

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten") 
model = VisionEncoderDecoderModel.from_pretrained("daekeun-ml/ko-trocr-base-nsmc-news-chatbot")
tokenizer = AutoTokenizer.from_pretrained("daekeun-ml/ko-trocr-base-nsmc-news-chatbot")

url = "https://raw.githubusercontent.com/aws-samples/aws-ai-ml-workshop-kr/master/sagemaker/sm-kornlp/trocr/sample_imgs/news_1.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

pixel_values = processor(img, return_tensors="pt").pixel_values 
generated_ids = model.generate(pixel_values, max_length=64)
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] 
print(generated_text)
```

All the code required for data collection and model training has been published on the author's Github.
- https://github.com/daekeun-ml/sm-kornlp-usecases/tree/main/trocr
