#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide name of conda virtual environment as an argument."
    echo "Usage example: ./create_conda_virtual_env.sh MyEnv"
    exit 1
fi

# Access the provided argument
argument=$1

# Rest of the script
echo "The provided conda virtual environment is: $argument"
# Add your desired script logic here

export VirtualEnv=$argument

conda create -y -n $VirtualEnv python=3.10.14 

source activate $VirtualEnv

pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=$VirtualEnv --display-name $VirtualEnv

pip install -r requirements.txt

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
