#!/bin/bash
#SBATCH -J torch-test    # create a short name for your job
#SBATCH -p gpu-share
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --mail-user=yhtsui@connect.ust.hk #Update your email address
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail

stdbuf -o0 -e0 command
module load anaconda3
source activate AT

python3 testCU.py 