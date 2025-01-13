# Anomaly Detection with Machine Learning

# 1. Problem statements
- **Detect anomal behavoir** of systems or machines with **fault explanations**

- - -

# 2. Methods
- **Random Cut Forest** (RCF) <br>
    * [Paper] https://assets.amazon.science/d2/71/046d0f3041bda0188021395b8f48/robust-random-cut-forest-based-anomaly-detection-on-streams.pdf
    * [Desc, KOREAN] https://hiddenbeginner.github.io/paperreview/2021/07/14/rrcf.html
    * [AWS Docs] https://docs.aws.amazon.com/kinesisanalytics/latest/sqlref/sqlrf-random-cut-forest.html  <br>
    * [AWS Sample codes] https://github.com/aws-samples/amazon-sagemaker-anomaly-detection-with-rcf-and-deepar
    

<p align="center">
    <img src="imgs/rcf-isolation.png" width="1100" height="300" style="display: block; margin: 0 auto"/>
</p>

- **RaPP** - Novelty Detection with Reconstruction along Projection Pathway <br>
    * [Ppaer, ICLR 2020] https://openreview.net/attachment?id=HkgeGeBYDB&name=original_pdf
    * [Desc, KOREAN] https://makinarocks.github.io/rapp/
    * [Supplement #1] [Autoencoder based Anomaly Detection](https://makinarocks.github.io/Autoencoder-based-anomaly-detection/)
    * [Supplement #2] [Reference code (github)](https://github.com/Aiden-Jeon/RaPP)
<p align="center">
    <img src="imgs/rapp-f1.png" width="1100" height="300" style="display: block; margin: 0 auto"/>
</p>
<p align="center">
    <img src="imgs/rapp-f2.png" width="1100" height="300" style="display: block; margin: 0 auto"/>
</p>


- - -

# 3. Results

<p align="center">
    <img src="imgs/rapp-explanation.png" width="1100" height="300" style="display: block; margin: 0 auto"/>
</p>
# anomaly-detection-with-explanation

# 4. Tutorials
- **Introduction to Anomaly Detection** <br>
    * [VOD] http://dmqm.korea.ac.kr/activity/seminar/339