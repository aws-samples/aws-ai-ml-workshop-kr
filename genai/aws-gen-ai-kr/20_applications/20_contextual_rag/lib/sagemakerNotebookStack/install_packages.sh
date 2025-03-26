#!/bin/bash

set -e

sudo -u ec2-user -i <<'EOF'
 
#source /home/ec2-user/anaconda3/bin/deactivate
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U pip
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U botocore>=1.35.72
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U requests>=2.31.0
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U requests-aws4auth>=1.2.0
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U opensearch-py>=2.3.0
/home/ec2-user/anaconda3/envs/python3/bin/python -m pip install -U python-dotenv>=1.0.0

EOF