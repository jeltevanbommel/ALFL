#!/bin/bash
#SBATCH --job-name=jupyter-notebook
#SBATCH --nodes=1
#SBATCH --cores-per-socket=6
#SBATCH --mem=16G
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="[titan-x|p100|gtx-1080ti]"
#SBATCH --output jupyter-notebook-%J.log

# load anaconda3
# module load anaconda3/2019.03

# for gpus support you definately need to load cuda
module load nvidia/cuda-10.2

echo "Gpu devices: "$CUDA_VISIBLE_DEVICES
# activate environment
#source activate conda-env

# Warning !
# Do not modify the sbatch script bolow this line !

#clean up XDG_RUNTIME_DIR variable
export XDG_RUNTIME_DIR=""

#Log Node Name
NODE=$(hostname)

# generate random port 8800-8809
PORT=$(((RANDOM % 10)+8800))

# start the notebook
for currround in {111..200}
do 
  for client in {0..8}
  do
   python3 research.py --action train --epochs 20 --reuse_weights --client "$client" --classes 2 --round "$currround" --name "$1"
  done
  python3 research.py --action fedavg --classes 2 --round "$currround" --name "$1"
done

# jupyter notebook --no-browser --ip=$NODE.ewi.utwente.nl --port=$PORT
