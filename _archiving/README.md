## AWS AI/ML Workshop - Korea

A collection of localized (Korean) AWS AI/ML workshop materials for hands-on labs. 

## Directory Structure

Hands-on materials wiil get enriched over time as we get more contributions. This repository has a number of distinct sub-directories for the below purpose:  

* Tested and verified lab materials are stored in `/src/release/YEAR-MM` sub directories. 
* `/src/work-in-progress` directory maintaines the latest work under development. You may test and contribute to make it stable enough to release. 
* `/doc/labguide` contains guidance materials for AWS users to follow up lab modules. This typically contains procedural step-by-step instructions with screenshots.
* `/images` is a collection of multimedia data used in lab modules. Any external images embedded in Jupyter notebooks should be stored here and referenced by `<img/>` tag within Jupyter Markdown cells.
* `/contribution` is where a user pushes their contribution to be included as a new lab module.

## Pre-requisites

This repository assumes that you have your own AWS account and want to experience [Amazon SageMaker](https://aws.amazon.com/sagemaker/). If you don't have an AWS account, please follow the below instruction.

* How to create a new Amazon Web Service account:
    * https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/
* Use the latest Chrome or Firefox browser.

## How To Use This Material?

* If you are attending AWS Korea's AI/ML workshop, session leaders will guide you the entire procedure.
    * For workshop participants, please check and register a credit if provided by session leaders.
* If you want to test this lab by yourself, we recommend to use the latest material stored in `/src/release/YEAR-MM` directory. 
    * Download the PDF file in `/src/release/YEAR-MM` directory. It includes a complete instruction on how to start your SageMaker notebook server, clone this repository into it, and perform the lab.
    * Please note that lab materials here may occur the charge depending on your choice of instance types and use of other AWS services. 
    * Each `/src/release/YEAR-MM` directory has README.MD file that explains the changes from the previous version. Read it carefully whether it meets your needs.

## Clean Up
To avoid incurring unnecessary charges, use the AWS Management Console to delete the resources that you created for this exercise. For the latest information, please refer [here](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-cleanup.html).

Note that if you plan to explore other exercises in this guide, you might want to keep some of these resources, such as your notebook instance, S3 bucket, and IAM role.

1. Open the Amazon SageMaker console at https://console.aws.amazon.com/sagemaker/ and delete the following resources:
    * The endpoint. This also deletes the ML compute instance or instances.
    * The endpoint configuration.
    * The model.
    * The notebook instance. You will need to stop the instance before deleting it.
1. Open the Amazon S3 console at https://console.aws.amazon.com/s3/ and delete the bucket that you created for storing model artifacts and the training dataset.
1. Open the IAM console at https://console.aws.amazon.com/iam/ and delete the IAM role. If you created permission policies, you can delete them, too.
1. Open the Amazon CloudWatch console at https://console.aws.amazon.com/cloudwatch/ and delete all of the log groups that have names starting with /aws/sagemaker/.

If you want to close your AWS account, please refer:
* [Korean version](https://docs.aws.amazon.com/ko_kr/awsaccountbilling/latest/aboutv2/close-account.html)
* [English version](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/close-account.html)

## How To Contribute Your Work

* If you want to make some changes, please create a branch and push your work for us to review separately.
* If you find typos or errors in Jupyter Notebook codes in existing notebooks under `/src/release/YEAR-MM`, please directly push your changes.
* This is same to all documents under `/doc/labguide`.
* When creating your notebook and if it is using any images, please put that images under `/images` directory. You may simply include the image in your Markdown cell in this way: `<img src="../../../images/Factorization2.png" alt="Factorization" style="width: 800px;"/>`
* Or else if you want to revise the content or propose a new lab, please push it to `/contribution`. We will review the content and decide whether to replace an existing module or to add it as a new module. The maintainer will accordingly modify or prepare a lab guide if your notebook contains good enough information for us to prepare a lab guide. Otherwise we will reach you back for further questions.
* For further details, please refer CONTRIBUTION.md

## References
* AWS AI/ML Blog: https://aws.amazon.com/blogs/machine-learning/
* Amazon SageMaker: https://aws.amazon.com/sagemaker/ 

## License Summary

This sample code is made available under a modified MIT license. See the LICENSE file.