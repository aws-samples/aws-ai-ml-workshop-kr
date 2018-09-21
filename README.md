## AWS AI/ML Workshop - Korea

A collection of localized Amazon SageMaker samples

## License Summary

This sample code is made available under a modified MIT license. See the LICENSE file.

## Directory structure

Hands-on materials get enriched over time as we get more contributions. This repository has a number of distinct sub-directories for the below purpose:  

* Tested and verified lab materials are stored in `/YEAR-MM` sub directories. 
* `/Work-in-progress` directory the latest work under development. You may test and contribute to make it stable enough to release. 
* `/LabGuide` contains a guidance material for AWS users to follow up lab modules. This typically contains procedural step-by-step instructions with screenshots
* `/images` is a collection of multimedia data used in lab modules. Any external images embedded in Jupyter notebooks should be stored here and referenced by `<img/>` tag within Jupyter Markdown cells.
* `/Contribution` is where a user pushes their contribution to be included as a new lab module.

### What to use for your AI/ML workshop

We recommend to use the latest material stored in `/YEAR-MM` directory. Each `/YEAR-MM` directory has README.MD file that explains the change from the previous version. Read it carefully if it meets with your needs.

## How to contribute your work

* For any changes, please create a branch and push your work for us to review separately.
* If you find typos or errors in Jupyter Notebook codes in existing notebooks under `/YEAR-MM`, please directly push your changes.
* This is same to all documents under `/LabGuide`.
* When creating your notebook and if it is using any images, please put that images under `/images` directory. You may simply include the image in your Markdown cell in this way: `<img src="../images/Factorization2.png" alt="Factorization" style="width: 800px;"/>`
* Or else if you want to revise the content or propose a new lab, please push it to `/Contribution`. We will review the content and decide whether to replace an existing module or to add it as a new module. The maintainer will accordingly modify or prepare a lab guide if your notebook contains good enough information for us to prepare a lab guide. Otherwise we will reach you back for further questions.
* For further details, please refer CONTRIBUTION.md