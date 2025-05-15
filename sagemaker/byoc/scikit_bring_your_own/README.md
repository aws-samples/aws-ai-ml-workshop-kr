## 도커이미지를 직접 작성하여 모델 훈련/추론 예시 

이 폴더는 Scikit-Learn 프레임워크를 이용한 커스텀 도커를 생성하고, 모델 훈련, 추론에 사용하는 코드 입니다.

- scikit_bring_your_own.ipynb 노트북을 실행하시면 됩니다.
    - 그리고 내부 작동을 이해하기 위해서 container 폴더 내용을 이해하시면 좋습니다.
    
- 아래는 scikit_bring_your_own.ipynb 노트북을 실행하고, 모델 서빙에 대한 CloudWatch Log 내용 입니다. 
```
Starting the inference server with 4 workers.
## After calling gunicorn server
[2025-05-15 00:54:03 +0000] [7] [INFO] Starting gunicorn 21.2.0
[2025-05-15 00:54:03 +0000] [7] [INFO] Listening at: unix:/tmp/gunicorn.sock (7)
[2025-05-15 00:54:03 +0000] [7] [INFO] Using worker: sync
[2025-05-15 00:54:03 +0000] [11] [INFO] Booting worker with pid: 11
[2025-05-15 00:54:03 +0000] [12] [INFO] Booting worker with pid: 12
[2025-05-15 00:54:03 +0000] [13] [INFO] Booting worker with pid: 13
[2025-05-15 00:54:03 +0000] [15] [INFO] Booting worker with pid: 15
[2025-05-15 00:54:09 +0000] [13] [INFO] ping start
[2025-05-15 00:54:09 +0000] [13] [INFO] get_model 
[2025-05-15 00:54:09 +0000] [13] [INFO] Model loaded successfully
[2025-05-15 00:54:13 +0000] [15] [INFO] ping start
[2025-05-15 00:54:13 +0000] [15] [INFO] get_model 
[2025-05-15 00:54:14 +0000] [15] [INFO] Model loaded successfully
[2025-05-15 00:54:18 +0000] [13] [INFO] ping start
[2025-05-15 00:54:18 +0000] [13] [INFO] get_model 
[2025-05-15 00:54:23 +0000] [11] [INFO] ping start
[2025-05-15 00:54:23 +0000] [11] [INFO] get_model 
[2025-05-15 00:54:24 +0000] [11] [INFO] Model loaded successfully
[2025-05-15 00:54:28 +0000] [11] [INFO] ping start
[2025-05-15 00:54:28 +0000] [11] [INFO] get_model 
[2025-05-15 00:54:33 +0000] [11] [INFO] ping start
[2025-05-15 00:54:33 +0000] [11] [INFO] get_model 
[2025-05-15 00:54:38 +0000] [12] [INFO] ping start
[2025-05-15 00:54:38 +0000] [12] [INFO] get_model 
[2025-05-15 00:54:39 +0000] [12] [INFO] Model loaded successfully
[2025-05-15 00:54:43 +0000] [15] [INFO] ping start
[2025-05-15 00:54:43 +0000] [15] [INFO] get_model 
[2025-05-15 00:54:48 +0000] [15] [INFO] ping start
[2025-05-15 00:54:48 +0000] [15] [INFO] get_model 
[2025-05-15 00:54:53 +0000] [15] [INFO] ping start
[2025-05-15 00:54:53 +0000] [15] [INFO] get_model 
[2025-05-15 00:54:58 +0000] [13] [INFO] ping start
[2025-05-15 00:54:58 +0000] [13] [INFO] get_model 
Invoked with 29 records
[2025-05-15 00:55:00 +0000] [13] [INFO] predict start
[2025-05-15 00:55:00 +0000] [13] [INFO] get_model 
[2025-05-15 00:55:00 +0000] [13] [INFO] Prediction complete, generated 29 predictions

```