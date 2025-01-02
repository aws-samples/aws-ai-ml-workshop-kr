# 설치 가이드

## 1. 실습환경
- 이 노트북은 [SageMaker Notebook Instance](https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html) 에서 테스트 완료 되었습니다.
    - 환경: ml.p3.2xlarge 및 생성시에 Additional configuration 클릭하고 200 GB EBS 추가 하였습니다. --> 참고: [Create a Notebook Instance](https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-create-ws.html)

## 2. 실습 환경 세팅
### 2.1. JupyerLab 을 열고 아래와 같이 터미널을 오픈 하세요.
- ![open_terminal.png](img/open_terminal.png)
### 2.2. 아래와 같이 명령어를 넣고 Git 리포를 클로닝 합니다.
    ```
    pwd
    cd SageMaker/
    git clone https://github.com/aws-samples/aws-ai-ml-workshop-kr.git
    ```
- ![git_clone.png](img/git_clone.png)
### 2.3. Conda Virtual Environment 생성 (약 7분 소요)
- 가상 환경을 생성시에 DeepSpeed 0.14.2 를 설치 합니다. 자세한 내용은 공식 [deepspeed 0.14.2 패키지 정보](https://pypi.org/project/deepspeed/)를 참조 하세요.
- setup 폴더로 이동을 합니다. 아래의 그림의 왼쪽 처럼, 탐색기로 setup 폴더까지 이동 합니다. 아래의 경로를 따라가 보세요.
    ```
    cd aws-ai-ml-workshop-kr/genai/aws-gen-ai-kr/30_fine_tune/04-deepspeed-on-sagemaker/setup
    ```
    - ![run_shell.png](img/run_shell.png)
- shell 을 아래의 명령어를 넣어서 실행 합니다. 위의 그림을 참조 하세요.
- 가상 환경 이름은 원하는 이름으로 하셔도 좋습니다. 여기서는 deepspeed-py310 으로 했습니다.
    ```
    ./create_conda_virtual_env.sh deepspeed-py310
    ```    
- 설치 확인을 합니다. 아래와 같은 화면이 나왔으면 성공입니다. 이렇게 나오지 않았으면 터미널에서 실행 내역에 "에러" 가 있는지 확인 하세요.
    - ![conda_install_result.png](img/conda_install_result.png)
- 콘다 가상 환경 (예: deepspeed-py310) 으로 들어가서 설치된 패키지를 확인 합니다.
- 아래와 같이 등러가지 못하면 `source ~/.bashrc` 를 실행하고 해보세요.
    ```
    conda env list
    source ~/.bashrc
    conda activate deepspeed-py310
    pip list | grep deepspeed
    ```
    - ![check_deepspeed_version.png](img/check_deepspeed_version.png)

### 2.4. 노트북 열고 커널 설정
- [notebook/01-sm-deepspeed-training.ipynb](../notebook/01-sm-deepspeed-training.ipynb) 을 열고 위에서 만든 콘다 커널을 설정 합니다. 여기서는 conda-deepspeed-py310 커널을 클릭 합니다.
- ![select_kernel.png](img/select_kernel.png)

### 2.5 설치가 끝났습니다. 축하 합니다.
- 이후는 노트북의 설명에 따라 실행 하시면 됩니다.






