#!/bin/bash
# Example for installing IPyWidgets extension from a SageMaker Lifecycle Configuration script
sudo -u ec2-user -i <<EOF
EXTENSION_NAME='@jupyter-widgets/jupyterlab-manager'
echo "Activating JupyterSystemEnv"
source /home/ec2-user/anaconda3/bin/activate JupyterSystemEnv
echo "Installing extension \$EXTENSION_NAME"
jupyter labextension install \$EXTENSION_NAME
echo "Deactivating JupyterSystemEnv"
source /home/ec2-user/anaconda3/bin/deactivate
EOF
