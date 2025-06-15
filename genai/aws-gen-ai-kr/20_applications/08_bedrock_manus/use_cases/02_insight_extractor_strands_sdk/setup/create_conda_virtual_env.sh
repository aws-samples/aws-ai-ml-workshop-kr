#!/bin/bash

# conda 초기화
eval "$(conda shell.bash hook)"

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
# conda create -y --prefix ./$VirtualEnv python=3.10.14
conda create -y -n $VirtualEnv python=3.12.9

# wait for  seconds
echo "# Wait for 5 seconds to proceed with next step"
sleep 5

#####################################
# Activate the given $VirtualEnv
#####################################
echo "## Enter virtual env" $VirtualEnv
source activate $VirtualEnv

# show current virtual env.
echo "# show current virtual env"
conda info --envs
which python
echo ""

# wait for  seconds
echo "## Finish creating virtual env"
sleep 5

#####################################
# Install Python package and Create ipykernel to show Kernel in Jupyter Notebook
#####################################

pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=$VirtualEnv --display-name "$VirtualEnv"

echo "## current folder and files"
pwd
ls

pip install -r requirements.txt
sh install_korean_font.sh
pip install browser-use==0.1.45
sudo apt-get install pandoc -y
sudo apt-get install texlive -y
sudo apt-get install texlive-xetex -y

# wait for  seconds
echo "## Finish installing requirements.txt"
sleep 5

#####################################
## Show usage command
#####################################
echo ""
echo "# Show important python packages"
pip list | grep -E "langchain|langgraph"
echo ""

echo "# To show conda env, use"
echo "#"
echo "# conda env list" 
echo "# To activate this environment, use"
echo "#"
echo "# source activate" $VirtualEnv
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

