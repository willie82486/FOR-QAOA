#!/bin/bash
#SBATCH --account=GOV113123
#SBATCH --partition=normal
#SBATCH -J QAOA_test
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --gpus-per-task=1
#SBATCH --gpus-per-node=1
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


#====================================================================
# BufferTuning - NVLink
#====================================================================
P=5
N=30
D=1
C=12
B=$((N-D))
cd ../../
make clean && make -j CHUNK_QUBIT=$C

output_file="2GPUs_2Node.log"
> "$output_file"

# for b in $(seq 10 "$B"); do
for b in $(seq 16 29); do
    echo "buffer size: $b" >> "$output_file"
    # mpirun -np 2 -bind-to none --map-by ppr:1:node nsys profile -o "profile/inter_node/inter_node_$b-%q{OMPI_COMM_WORLD_RANK}" --force-overwrite true ./weighted "$P" "$N" "$D" "$C" "$b" >> "$output_file"
    mpirun -np 2 -bind-to none --map-by ppr:1:node ./weighted "$P" "$N" "$D" "$C" "$b" >> "$output_file"
done
