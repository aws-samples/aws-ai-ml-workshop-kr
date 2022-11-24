#!/bin/python

import subprocess
import sys

if __name__ == "__main__":
    arguments_command = " ".join([arg for arg in sys.argv[1:]])
    """
    The following line takes care of setting up an inter-node communication
    as well as managing intra-node workers for each GPU.
    """
    subprocess.call("python -m torch_xla.distributed.sm_dist " + arguments_command, shell=True)