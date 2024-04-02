
# Bedrock Demo
This repository is a Gen AI demo using claude 3. It contains 5 pre-built examples which help customers getting started with the Amazon Bedrock.

![Pic 1.](architecture.png)

## Prerequisites
- Make sure you have [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) installed and configured with an aws account you want to use.
- To build container image, [Docker Engine](https://docs.docker.com/engine/install/) must be installed.

### install cdk
```shell
npm install -g aws-cdk
cdk --version
```

### setting AWS_PROFILE
```shell
export AWS_PROFILE=[The configuration profile for aws-cli]
```

## How to deploy

### Step 1. Create virtualenv
After cloning this git repository. You need to create virtualenv.
```shell
cd bedrock-demo
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2. Install requirements
```shell
pip install -r requirements.txt
```

### Step 3. Deploy CDK
Before running the cdk deploy command, make sure the Docker Engine is running in the background.
```shell
cdk deploy BedrockDemo
```

### Step 4. Set Environment Variables
After successfully deploying the cdk, there are two variables in the output. One is the DNS name of the ALB and the other is the name of the S3 bucket. Copy these and make them environment variables.

![Pic 2.](cdk_output.png)


```shell
export ALB_URL=[The DNS name of the ALB]
export BUCKET_NAME=[The name of the S3 bucket]
```

### Step 5. Run streamlit
```shell
cd frontUI
streamlit run app/Home.py
```

![Pic 3.](run_streamlit.png)

The browser will open a demo page if streamlit is running successfully. 

## Clean up resources
If you've created anything by yourself, you'll need to delete it.

### Destroy Stack
```shell
cdk destroy BedrockDemo
```
