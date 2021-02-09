# willettlabs_pytorch_tutorial

This tutorial is intended to provide a brief walkthrough of how I use PyTorch, and maybe give some tips and
useful scripts to help get you started.

This tutorial is not intended to be an introduction to neural networks, Python, or how to use the Slurm
cluster management system. There are bits and pieces of those things in here, though.

## Dependencies

The real prerequisites here are git and conda. All Python used will be Python 3, and all Pytorch used will be >= 1.0.
I will attempt to avoid anything being super specific to versions. However, since deep learning frameworks love 
changing things in backwards-incompatible ways, I can make no promises that the code here will run on the latest
version of Pytorch forever.

### Installing conda

Installing conda and always working in virtual environments/conda environments is highly recommended.
In fact, if you don't already do that, please let this be your one takeaway from this tutorial. The ability to 
destroy a busted environment without worrying you're removing your native Python installation is great.
Not messing up all your code from 2018 because you updated to a new version of PyTorch is invaluable.

To install conda on a Mac or Windows device:

Go here: [Anaconda Link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)<br>
Follow the instructions. I'd recommend Anaconda, but miniconda is fine for us.

To install conda on your slurm account:

Go here: [Miniconda source](https://docs.conda.io/en/latest/miniconda.html)
and download the latest Miniconda package.<br>
If you're using a Mac or Linux device, open Terminal. Navigate to your Downloads, and `scp` the downloader to Slurm.<br>
Then ssh to the cluster, and run the script.

This can be done with the following commands (Change my username in these lines to yours!):<br>
    
    cd Downloads
    scp Miniconda3-latest-Linux-x86.sh gilton@slurm.ttic.edu:~
    ssh gilton@slurm.ttic.edu
    bash Miniconda3-latest-Linux-x86.sh



### Installing necessary packages

Package management is important. We use conda rather than pip because conda plays really nicely with installing
gpu-capable versions of Pytorch.

Most of the following commands will still work even if you don't have a GPU, don't worry! I'll note the things
 to change if you don't have a GPU.
 
Navigate your terminal to this repository's base directory, and then run:


    conda env create -f environment.yml

    
Respond ``y`` to any questions. To activate this environment, type


    conda activate simple_pytorch_environment

    
Make sure you can run Python, and are able to run ``import torch``



### Installing necessary packages on Slurm
If you're running this on Slurm, do the following. **Note: Using conda to install pytorch will NOT install the
GPU-compatible version if you run this script using the head node! Submit a job!**

First, clone this repository to your home directory. Then, create a text file named conda_install with the following lines in it:


    #!/bin/bash
    
    source activate  
    conda env create -f ~/willettlabs_pytorch_tutorial/environment.yml &>> conda_installation.txt

    
Execute this script **on a gpu-enabled node** with the following:

    sbatch -p willett-gpu -c1 ~/conda_install
    
    

## Checkpoints
You can find the checkpoint used in the single_image_testing.py demo on Box [here](https://uwmadison.box.com/s/w1jd55bnk13v2s525vsh1bc70z9elvs3)

## Acknowledgments
The code used in this repository is based on the code for [Deep Back-Projection Networks for Single-Image Superresolution](https://github.com/alterzero/DBPN-Pytorch).

