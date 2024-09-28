
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

# Get TGI image
docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.3.0-tgi2.0.2-gpu-py310-cu121-ubuntu22.04 

# export env. vars 
export MODEL_REPO_DIR=/home/ec2-user/SageMaker/models/llama-3-8b-naver-news-s3-download
export MODEL_LOG_DIR=/home/ec2-user/SageMaker/tmp/log

mkdir -p $MODEL_LOG_DIR

docker run -it --runtime=nvidia --gpus all --shm-size 12g \
 -v $MODEL_REPO_DIR:/opt/ml/model:ro \
 -v $MODEL_LOG_DIR:/opt/djl/logs \
 -p 8080:8080 \
 -e HF_MODEL_ID=/opt/ml/model \
 -e SM_NUM_GPUS=4 \
 -e MESSAGES_API_ENABLED=true \
 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.3.0-tgi2.0.2-gpu-py310-cu121-ubuntu22.04 

# if needed, comment out and run as a test
# curl http://localhost:8080/generate \
#     -X POST \
#     -d '{"inputs":"What is the capital of France?","parameters":{"max_new_tokens":50}}' \
#     -H 'Content-Type: application/json'

# curl http://localhost:8080/generate_stream \
#     -X POST \
#     -d '{"inputs":"Tell me a short story","parameters":{"max_new_tokens":100}}' \
#     -H 'Content-Type: application/json'