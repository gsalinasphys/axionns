#!/bin/bash
####################################
#   ARIS slurm script template  #
#                 #
# Submit script: sbatch filename  #
#                 #
####################################

#SBATCH -J AxionNS
#SBATCH -t 01:00:00
#SBATCH -A cops    # Account
#SBATCH -p cops    # Partition
#SBATCH --ntasks=64
#SBATCH --output=log.AxionNS.out # Stdout (%j expands to jobId)
#SBATCH --error=error.AxionNS.err # Stderr (%j expands to jobId)

module load conda
python run.py
