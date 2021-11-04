## AutoML with AutoGluon, Amazon SageMaker, and AWS Lambda

본 리포지토리의 예제는 다음 블로그 포스트의 예제를 한글화하고 일부코드를 수정하고 한글화한 것입니다.
- [Code-free machine learning: AutoML with AutoGluon, Amazon SageMaker, and AWS Lambda](https://aws.amazon.com/blogs/machine-learning/code-free-machine-learning-automl-with-

워밍업 단계로 AutoGluon의 사용방법을 이해하기 위해서는 다음 노트북을 실행해 보세요.
- [AutoGluon Hello World! - autogluon_helloworld.ipynb](autogluon_helloworld.ipynb)  

Code-free AutoGluon 예제의 실행은 다음 노트북을 참고하세요.
- [Code-free AutoGluon - codefree_autogluon.ipynb](codefree_autogluon.ipynb)


---

This repository contains the CloudFormation template and prewritten source code powering the code-free AutoML pipeline detailed in [this AWS Machine Learning blog post](https://aws.amazon.com/blogs/machine-learning/code-free-machine-learning-automl-with-autogluon-amazon-sagemaker-and-aws-lambda/). Feel free to customize it to fit your use case and share with us what you build!

* `autogluon-tab-with-test.py` is the script run by the SageMaker training job the Lambda function automatically kicks off when you upload your training data to S3. It's pre-packaged in `sourcedir.tar.gz` for the use of the pipeline. You can modify this script to reuse the pipeline with your own model training code.
* `CodeFreeAutoML.yaml` is the CloudFormation template you use to deploy the pipeline in your account.
* `lambda_function.py` is the source code for the Lambda function that kicks off the SageMaker training job when you upload your data to S3.
* `sourcedir.tar.gz` is the `autogluon-tab-with-test.py` file pre-packaged for your convenience; the pipeline requires it to be gzipped.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

