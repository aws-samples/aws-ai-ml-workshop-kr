#!/bin/bash

#####################################
# Check if an argument is provided
#####################################
if [ $# -eq 0 ]; then
    echo "Error: Please provide name of conda virtual environment as an argument."
    echo "Usage example: ./create_conda_virtual_env.sh MyEnv"
    exit 1
fi

#####################################
# Access the provided argument
#####################################
argument=$1

# Rest of the script
echo "The provided conda virtual environment is: $argument"
# Add your desired script logic here

export VirtualEnv=$argument

#####################################
# Create conda virtual env.
#####################################
conda create -y -n $VirtualEnv python=3.10.14

# wait for  seconds
echo "# Wait for 5 seconds to proceed with next step"
sleep 5

#####################################
# Activate the given $VirtualEnv
#####################################
source activate $VirtualEnv

# show current virtual env.
echo "# show current virtual env"
conda info --envs
which python
echo ""

# wait for  seconds
echo "## Wait for seconds to proceed with next step"
sleep 5

#####################################
# Install Python package and Create ipykernel to show Kernel in Jupyter Notebook
#####################################

pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=$VirtualEnv --display-name "$VirtualEnv"

pip install -r requirements.txt

# wait for  seconds
echo "## Wait for seconds to proceed with next step"
sleep 5

#####################################
## install or upgrade gcc to 10.2.1
#####################################
echo "#########################################"
echo "# Upgrading gcc"
echo "#########################################"

echo "# Current gcc version"
gcc --version
echo ""
sudo yum update -y
sudo yum install -y centos-release-scl
sudo yum install -y devtoolset-10-gcc devtoolset-10-gcc-c++
scl enable devtoolset-10

source /opt/rh/devtoolset-10/enable
echo ""
echo "# After gcc version"
gcc --version
echo ""

echo "#########################################"
echo "# End of Upgrading gcc"
echo "#########################################"

#####################################
## Show usage command
#####################################
echo ""
echo "# Show important python packages"
pip list | grep -E "datasets|transformers|fsspec|evaluate|deepspeed|s3fs|boto3|sagemaker|scikit-learn"
echo ""

echo "# To show conda env, use"
echo "#"
echo "# conda env list" 
echo "# To activate this environment, use"
echo "#"
echo "# conda activate" $VirtualEnv
echo "#"
echo "# To deactivate an active environment, use"
echo "#"
echo "# $ conda deactivate"
echo "#"
echo "# To remove an active environment, use"
echo "# conda env remove -n" $VirtualEnv
echo "# Show jupyter kernels"
echo "# jupyter kernelspec list"
echo "# Remvoe the given jupyter kernels"
echo "# jupyter kernelspec uninstall -y " $VirtualEnv
echo ""

# Apply this setting
source ~/.bashrc 

