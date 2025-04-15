#!/bin/bash
#SBATCH --job-name=my_little_job  # Job name
#SBATCH --time=72:00:00       	  # Time limit hrs:min:sec
#SBATCH -N 1            	        # Number of nodes
#SBATCH -w gorgona6
#SBATCH --mail-type=ALL
#SBATCH --mail-user=my_mail@mail.com

set -x # all comands are also outputted

cd /home/all_home/josefernandes

module list
module avail
module load python3.10.12

source ./miniconda3/bin/activate
conda activate mimicbaseline

cd /home_cerberus/speed/josefernandes/
cd strodthoff2024prospects
# export HF_HOME=/home_cerberus/disk2/josefernandes/huggingface/
# export TOKENIZERS_PARALLELISM=false
cd src
python3 h5_preprocessing.py

hostname   # just show the allocated node
