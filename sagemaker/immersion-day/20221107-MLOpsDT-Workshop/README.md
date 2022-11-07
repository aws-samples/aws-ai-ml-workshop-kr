# 2022 AWS Seoul MLOps 및 분산학습 워크샵 실습 가이드
- 날짜
    - 2022.11.08 ~ 11.09 (화, 수)
- 참조 : 메인 이벤트 페이지
    - [MLOps 및 분산학습 워크샵 일정](https://mlops-distributed-training-korea-2022.splashthat.com/)
    
<br>

# 1. 실습 환경 구성
- 이벤트 엔진 환경 구성 가이드 링크
    - TBD

## 1.1. Day1 ML Ops
- ML Ops 실습 해시 코드
    - TBD
- SageMaker Studio 기본 생성 사용

## 1.2. Day2 분산 학습
- 분산 학습 실습 해시 코드
    - us-east-1
    - us-west-2
    - eu-west-1: <b>`6a26-157b8db814-cd`</b>
- SageMaker Notebook <b>`ml.m5.xlarge`</b> 로 생성

<br>

# 2. Day1: ML Ops 실습 상세
## 2.1. Day1 ML Ops 실습 링크 및 내용
- TBD

<br>

# 3. Day2: 분산 학습 실습 상세

## 3.1. Day2 분산 학습 실습 링크 및 내용
- Git Repo:
    - Distributed Training Workshop on Amazon SageMaker
- URL: 
    - https://github.com/aws-samples/sagemaker-distributed-training-workshop
- 실습 내용
    - ![dt-workhop-labs](img/dt-workhop-labs.png)

<br>

## 3.2. Lab01 : Amazon SageMaker 기반 데이터 병렬화(Data Parallelism) 실습

### 3.2.1. Lab01 코드 개요 (`Lab_1.ipynb`)
#### (1) pytorch-lightning 설치를 위해서 `requirements.txt` 생성
```python
%%writefile scripts/requirements.txt
pytorch-lightning == 1.6.3
lightning-bolts == 0.5.0
```
#### (2) 모델 훈련 코드 준비 (`mnist.py`)
```python
# 파이토치 라이트닝 “환경” 준비
from pytorch_lightning.plugins.environments.lightning_environment import LightningEnvironment
env = LightningEnvironment()
env.world_size = lambda: int(os.environ.get("WORLD_SIZE", 0))
env.global_rank = lambda: int(os.environ.get("RANK", 0))

# 파이토치 라이트닝 Trainer 에 strategy=ddp 와 함께 생성.
import pytorch_lightning as pl
trainer = pl.Trainer(max_epochs=args.epochs, strategy=ddp, devices=num_gpus, num_nodes=num_nodes, default_root_dir = args.model_dir)

# 파이토치 라이트닝 Trainer 에 MNIST 모델 및 데이타 제공하여 훈련
trainer.fit(model, datamodule=dm)
```

#### (3) AWS Deep Learning Container (DLC)  준비
```python
image_uri = '763104351884.dkr.ecr.{}.amazonaws.com/pytorch-training:1.12.0-gpu-py38-cu113-ubuntu20.04-sagemaker'.format(region)
```
#### (4) SageMaker Estimator 생성
```python
estimator = PyTorch(
  entry_point="mnist.py",
  base_job_name="lightning-ddp-mnist",
  image_uri = image_uri,
  role=role,
  source_dir="scripts",
  # configures the SageMaker training resource, you can increase as you need
  instance_count=1,
  instance_type=instance_type,
  py_version="py38",
  sagemaker_session=sagemaker_session,
  distribution={"pytorchddp":{"enabled": True}},
  debugger_hook_config=False,
  # profiler_config=profiler_config,
  hyperparameters={"batch_size":32, "epochs": epoch},
  # enable warm pools for 20 minutes
  keep_alive_period_in_seconds = 20 *60
)
```

#### (5) SageMaker Estimator 의 fit() 함수 실행
```python
# Passing True will halt your kernel, passing False will not. Both create a training job. 
estimator.fit(wait=False)
```

#### (6) 모델 학습 Job 생성 및 훈련 실행
![training-job-log.jpg](img/training-job-log.jpg)

#### (7) 멀티 노드 분산 훈련 (예: `ml.g4dn.12xlarge` 2대 사용)
- estimator 에 instance_count 의 실행할 노드 수만 제공

```python
estimator = PyTorch(
  ## instance_count = 2 로 설정시 2개의 노드 (예: ml.g4dn.12xlarge 2대)
  instance_count=2,
)
```
- 모델 훈련 Job 의 실행 로그를 보면 2개의 노드가 훈련을 실행 함.
![two-node-dt.jpg](img/two-node-dt.jpg)

#### (8) S3에 모델 아티펙트 (가중치 파일) 생성
- 위치: 
    - SageMaker Console -> 왼쪽 메뉴의 Training -> Training Jobs -> 해당 Training Jobs 클릭 -> 아래쪽에 Output 섹션 확인  
![training-artifact.png](img/training-artifact.png)

### 3.2.2. Demo Video on YouTube
- [PyTorch DDP on SageMaker Distributed Data Parallel](https://www.youtube.com/watch?v=0VWENkgPuYg)


<br>

## 3.3. Lab02 : Amazon SageMaker 기반 모델 병렬화(Model Parallelism) 실습

### 3.3.1. Lab02 코드 개요 (`Lab_2.ipynb`)

#### (1) 모델 훈련 코드 준비
- 스크립트 개요
    - `train_gpt_simple.py`: The entrypoint script passed to the Hugging Face estimator in this notebook. This script is responsible for end to end training of the GPT-2 model with SMP. You can follow the comments to learn where the SMP API is used.
    - `data_pipeline.py`: 훈련 데이터를 준비하기 위한 Datapipeline 함수.
    - `data_prep_512.py`: openwebtext 데이터셋을 다운로드 및 전처리
    - `learining_rate.py`: 학습률 조정
    - `requirements.txt`: 허깅페이스 transformers 라이브러리를 비롯한 의존성 패키지 설치
    - `memory_tracker.py`: 메모리 사용량 추정
    - `sharded_data_parallel_checkpoint.py`: 분할 데이터 병렬화를 위한 체크포인트 유틾리티 함수

- 파이프라인 병렬화를 위한 `@smp.step` 데코레이터 정의
```python
@smp.step
def train_step(model, optimizer, input_ids, attention_mask, args):
    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)["loss"]
    model.backward(loss)
    return loss

@smp.step
def test_step(model, input_ids, attention_mask):
    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)["loss"]
    return loss
```

- 텐서 병렬화 설정
```python
model = smp.DistributedModel(model, trace_device="gpu", backward_passes_per_step=args.gradient_accumulation)
...
optimizer = smp.DistributedOptimizer(
        optimizer, 
        static_loss_scale=None, 
        dynamic_loss_scale=True,
        dynamic_loss_args={"scale_window": 1000, "min_scale": 1, "delayed_shift": 2},
        )
```

#### (2) AWS Deep Learning Container (DLC)  준비
```python
image_uri = f'763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:1.12.0-gpu-py38-cu113-ubuntu20.04-sagemaker'
```

#### (3) (Optional) FSx for Lustre 설정
```python
if use_fsx:
    from sagemaker.inputs import FileSystemInput

    # Specify FSx Lustre file system id.
    file_system_id = ""

    # Specify the SG and subnet used by the FSX, these are passed to SM Estimator so jobs use this as well
    fsx_security_group_id = ""
    fsx_subnet = ""

    # Specify directory path for input data on the file system.
    # You need to provide normalized and absolute path below.
    # Your mount name can be provided by you when creating fsx, or generated automatically.
    # You can find this mount_name on the FSX page in console.
    # Example of fsx generated mount_name: "3x5lhbmv"
    base_path = ""

    # Specify your file system type.
    file_system_type = "FSxLustre"

    train = FileSystemInput(
        file_system_id=file_system_id,
        file_system_type=file_system_type,
        directory_path=base_path,
        file_system_access_mode="rw",
    )

    data_channels = {"train": train, "test": train}
```

#### (4) SageMaker Estimator 생성
```python
smp_estimator = PyTorch(
    entry_point="train_gpt_simple.py",
    source_dir=os.getcwd(),
    role=role,
    instance_type=instance_type,
    volume_size=volume_size,
    instance_count=instance_count,
    sagemaker_session=sagemaker_session,
    distribution={
        "mpi": {
            "enabled": True,
            "processes_per_host": processes_per_host,
            "custom_mpi_options": mpioptions,
        },
        "smdistributed": {
            "modelparallel": {
                "enabled": True,
                "parameters": {
                    "ddp": True,
                    "skip_tracing": True,
                    "delayed_parameter_initialization": hyperparameters["delayed_param"] > 0,
                    "offload_activations": hyperparameters["offload_activations"] > 0,
                    "activation_loading_horizon": hyperparameters["activation_loading_horizon"],
                    "sharded_data_parallel_degree": hyperparameters["sharded_data_parallel_degree"],
                    "fp16": hyperparameters["fp16"] > 0,
                    "bf16": hyperparameters["bf16"] > 0,
                    # partitions is a required param in the current SM SDK so it needs to be passed,
                    "partitions": 1,
                },
            }
        },
    },
    framework_version="1.12",
    py_version="py38",
    output_path=s3_output_location,
    checkpoint_s3_uri=checkpoint_s3_uri if not use_fsx else None,
    checkpoint_local_path=hyperparameters["checkpoint-dir"] if use_fsx else None,
    metric_definitions=metric_definitions,
    hyperparameters=hyperparameters,
    debugger_hook_config=False,
    disable_profiler=True,
    base_job_name=base_job_name,
    **kwargs,
)
)
```

#### (5) SageMaker Estimator 의 fit() 함수 실행
```python
# Passing True will halt your kernel, passing False will not. Both create a training job.
smp_estimator.fit(inputs=data_channels, logs=True)
```
### 3.3.2. Demo Video on YouTube
- [GPT-2 training using Amazon SageMaker Model Parallelism Library](https://youtu.be/TwkLh4QMTmc)
- [(Optional) Amazon FSx for Lustre setup Demo](https://youtu.be/oxRxW6qXDKI)
- [(Optional) Mask R-CNN Pre-training Demo w/ Amazon FSx for Lustre](https://youtu.be/oYhre0Ci9QM)

<br>

# References

## 분산 학습 개발자 가이드 및 참조 자료
- 분산 학습 개발자 가이드
    - [Amazon SageMaker Distributed Training Libraries](https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-training.html)
    - ![sm-developer-guide.png](img/sm-developer-guide.png)
- SageMaker Distributed Data Parallel (SM DDP)
    - [Supported Frameworks, AWS Regions, and Instances Types](https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-data-parallel-support.html)
    - ![sm-ddp-framework.png](img/sm-ddp-framework.png)
- SageMaker Distributed Model Parallel (SM DMP)
    - [Supported Frameworks, AWS Regions, and Instances Types](https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-model-parallel-support.html)