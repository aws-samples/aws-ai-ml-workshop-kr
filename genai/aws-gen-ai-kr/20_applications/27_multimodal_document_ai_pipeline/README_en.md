# PDF QA Extraction

[English](README_en.md) | [한국어](README.md)

This tool leverages GPU acceleration to extract text blocks from PDF documents and uses Amazon Bedrock's Claude model to automatically generate high-quality question-answer pairs from the extracted content. Through this process, document knowledge is transformed into structured QA JSON datasets that can be used for training, fine-tuning, or knowledge base construction.

[PDF QA Extraction Process Video Guide](https://assets.fsi.kr/videos/qna-extract.mp4)

## System Flow Diagram

![GPU Container Process](../assets/images/flow.png)

*The above diagram shows the complete process of extracting QA data from PDF documents. When a PDF document is input, it is converted into text blocks through the Unstructured partition extractor, and this data is processed into structured JSONL QA data using the Claude LLM.*

## Installation Guide

### Building Unstructured CUDA Docker Image

Unstructured provides powerful tools for extracting and processing content from PDFs. This tool performs block-level text extraction from documents to convert data into structured format. To set up the Docker environment, follow these steps:

1. Ensure Docker is installed on your system.

2. Build Docker image:
     ```bash
     docker build -t qa-extractor -f Dockerfile .
     ```

     ```bash
     # Event Engine lab accounts have network restrictions and must use this Dockerfile_event_eng.
     docker build -t qa-extractor -f Dockerfile_event_eng .     
     ```

### GPU-based PDF Extractor Usage Guide

#### 1. Running in Local GPU Environment

The Unstructured Extractor uses GPU to quickly extract text from PDF documents. The Docker container supports NVIDIA GPUs and can be run as follows:

```bash
# Linux/macOS
docker run --rm --gpus all -v $(pwd):/app -w /app --env-file .env qa-extractor python processing_local.py

# Windows
docker run --rm --gpus all -v %cd%:/app -w /app --env-file .env qa-extractor python processing_local.py
```

To verify GPU support is enabled:
```bash
docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
```

#### Environment Variable Configuration

The following environment variables must be set during execution:
- `AWS_REGION`: AWS region (e.g., us-east-1)
- `PDF_PATH`: Path to the PDF file to process
- `DOMAIN`: Document subject domain (e.g., "International Finance")
- `NUM_QUESTIONS`: Number of questions to generate per text element
- `NUM_IMG_QUESTIONS`: Number of questions to generate per image
- `MODEL_ID`: Bedrock model ID to use (e.g., anthropic.claude-3-sonnet-20240229-v1:0)
- `TABLE_MODEL`: Table structure inference model (e.g., yolox)

## Table Extraction Model Comparison

### Detailed Performance Comparison

| Model | Vendor | Accuracy | Speed | GPU Memory | Features |
|-------|--------|----------|-------|------------|----------|
| detectron2 | Meta | ⭐⭐⭐⭐⭐ | ⭐⭐ | High | Highest accuracy, research use |
| detectron2_onnx | Meta | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Medium | ONNX optimized version |
| table-transformer | Microsoft | ⭐⭐⭐⭐⭐ | ⭐⭐ | High | Excellent for complex tables, Download issue in SageMaker Processing (2025-09-08) |
| tatr | Community | ⭐⭐⭐⭐ | ⭐⭐⭐ | Medium | Balanced performance |
| yolox | Megvii | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Low | Fast processing |
| yolox_quantized | Megvii | ⭐⭐ | ⭐⭐⭐⭐⭐ | Very Low | Ultra-fast processing |
| paddle | Baidu | ⭐⭐⭐ | ⭐⭐⭐ | Medium | Chinese table specialized |
| chipper | Community | ⭐⭐ | ⭐⭐⭐⭐ | Low | Lightweight model |

### Recommended Models by GPU Memory

| GPU Memory | Recommended Model | Features |
|------------|-------------------|----------|
| Under 4GB | yolox_quantized | Ultra-lightweight |
| 4-8GB | yolox | Basic |
| 8-12GB | tatr | Balanced |
| 12-16GB | table-transformer | High performance |
| 16-24GB | detectron2_onnx | Meta optimized |
| 24GB+ | detectron2 | Meta highest performance |

### Recommended Models by Use Case

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| Research paper analysis | detectron2 | Highest accuracy required |
| Reports | table-transformer | Many complex tables |
| General documents | tatr | Balanced choice |
| Real-time processing | yolox_quantized | Speed priority |
| Batch processing | detectron2_onnx | Large-scale processing |
| Chinese documents | paddle | Language specialized |
| IoT Edge | chipper | Lightweight |

```bash
# Create .env file
touch .env

# Open .env file with vi editor
vi .env
```

Press i to enter input mode
Copy and paste the content below:

```bash
# App Setting
PDF_PATH=data/fsi_data.pdf
DOMAIN=International Finance
NUM_QUESTIONS=5
NUM_IMG_QUESTIONS=1
MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
TABLE_MODEL=yolox

# AWS Configuration 
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_SESSION_TOKEN=your_session_token_here

# Press ESC and type :wq to save and exit
```

> **Note**: Use the `.env` file only for local testing purposes. Using IAM roles in production environments is the reference architecture practice. Please be careful not to expose AWS Keys externally.

#### Performance Optimization Tips

- Large PDF files (over 100MB) should be split before processing
- Running in an environment with CUDA-compatible GPU improves processing speed by 5-10 times
- Monitor memory usage and adjust the `batch_size` parameter if necessary (refer to partition_pdf in the code)

#### 2. Running on SageMaker Processing Job

The Unstructured-qa-extractor image can be run as a batch job through Amazon SageMaker Processing Jobs:

1. Push image to ECR:
    The following commands in the terminal perform ECR authentication, image tagging, repository creation, and image push processes respectively. They register the locally built Docker image to AWS ECR so it can be used in SageMaker.
     ```bash
     # ECR login - Perform AWS authentication
     aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.<your-region>.amazonaws.com
     # Tag local image for ECR
     docker tag qa-extractor <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/qa-extractor
     # Create ECR repository
     aws ecr create-repository --repository-name qa-extractor --region <your-region>
     # Push image to ECR
     docker push <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/qa-extractor
     ```

2. Create SageMaker Processing Job:

     SageMaker Processing Job is a feature of AWS SageMaker for handling various stages of ML workflows such as data preprocessing, postprocessing, and model evaluation.
     For detailed examples on creating Unstructured Q&A Processing Jobs, refer to the `sagemaker_processingjob_pdf_qa_extraction.ipynb` notebook.
     
          ```python
          from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor

          # Create processor object
          processor = Processor(
              role='your-iam-role',
              image_uri='your-container-image',
              instance_count=1,
              instance_type='ml.g5.xlarge',
              volume_size_in_gb=30
          )

          # Run processing job
          processor.run(
              inputs=[
                  ProcessingInput(
                      source='s3://your-bucket/input-data',
                      destination='/opt/ml/processing/input'
                  )
              ],
              outputs=[
                  ProcessingOutput(
                      source='/opt/ml/processing/output',
                      destination='s3://your-bucket/output-data'
                  )
              ],
              code='path/to/your/processing_script.py'
          )
          ```
          
          **Processing Job Configuration Explanation:**
          - `role`: IAM role ARN for SageMaker to access AWS resources
          - `image_uri`: qa-extractor container image URI uploaded to ECR
          - `instance_count`: Number of instances to run (increase for parallel processing)
          - `instance_type`: GPU instance type to use for processing job
          - `volume_size_in_gb`: EBS storage volume size allocated for processing job
          - `inputs`: Specify data path to import from S3 bucket to container (/opt/ml/processing/ is default)
          - `outputs`: Specify S3 path to store processing results (/opt/ml/processing/ is default)
          - `code`: Path to processing script to run inside container
          
This approach allows efficient management and scaling of large-scale PDF processing tasks.

## Usage

This directory contains scripts for:
- PDF text extraction
- Content processing
- Question-answer pair generation

Please refer to individual script documentation for detailed usage of each tool.

## Dependencies

- Python 3.8+
- Unstructured GPU TEXT Extractor Image 
- SageMaker Processing Job

## 👥 Contributors
- **HyeonSang Jeon** (AWS Solutions Architect) | [Mail](mailto:hsjeon@amazon.com) | [Git](https://github.com/hyeonsangjeon) |

- - -

## 🔑 License
- This is licensed under the [MIT License](https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/LICENSE).