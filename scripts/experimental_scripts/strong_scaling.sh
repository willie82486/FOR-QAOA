#!/bin/bash
#SBATCH --account=GOV113123
#SBATCH --partition=normal
#SBATCH -J QAOA_test
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus=4
#SBATCH --gpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH -o out_multi_node.log
#SBATCH -e err_multi_node.log

module purge
module load cuda/12.2 
module load nvhpc/24.7
# module load singularity

export UCX_NET_DEVICES=mlx5_0:1
export UCX_IB_GPU_DIRECT_RDMA=1
export MPI_HOME=/work/HPC_software/LMOD/nvidia/packages/hpc_sdk-24.7/Linux_x86_64/24.7/comm_libs/mpi/
export NCCL_ROOT=/work/HPC_software/LMOD/nvidia/packages/hpc_sdk-24.7/Linux_x86_64/24.7/comm_libs/nccl/

# export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_EXT=1
export NCCL_SOCKET_IFNAME=ib0



P=5
N=30
D=2
C=12
B=23
cd ../../
make clean && make -j CHUNK_QUBIT=$C

device_num=$((2 ** D))
output_file="strong_scaling/strong_scaling_device_"$device_num".log"
> "$output_file"


echo "device_num : $device_num" >> "$output_file"
# nsys profile -o "single_device" --force-overwrite true mpirun -np $device_num -bind-to none --map-by ppr:8:node ./weighted "$P" "$N" "$D" "$C" "$B" >> "$output_file"
# ncu -o "single_device" -f mpirun -np $device_num -bind-to none --map-by ppr:8:node ./weighted "$P" "$N" "$D" "$C" "$B" >> "$output_file"
mpirun -np $device_num -bind-to none --map-by ppr:8:node ./weighted "$P" "$N" "$D" "$C" "$B" >> "$output_file"
