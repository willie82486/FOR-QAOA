#!/bin/bash
#SBATCH --account=GOV113123
#SBATCH --partition=normal
#SBATCH -J QAOA_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH -o out_single.log
#SBATCH -e err_single.log

P=5
D=0
C=11
export HYQUAS_ROOT=~/FOR-QAOA-GPU/HyQuas/
echo "Running QuEST"
cd ~/FOR-QAOA-GPU/QuEST/QuEST/
cmake -B build -DUSER_SOURCE=../QAOA_QuEST.c -DGPUACCELERATED=1 -DGPU_COMPUTE_CAPABILITY=89
cd build && make
./demo 5

echo "Running Aer"
cd ~/FOR-QAOA-GPU/
for Q in {26..27};
do  
    python3 aer.py --fusion_enable ./circuit_qaoa/qasm/${Q}/qaoa${Q}.qasm ${D}
done 

echo "Running cuQuantum"
cd ~/FOR-QAOA-GPU/
for Q in {26..27};
do  
    python3 aer.py --fusion_enable --cuStateVec_enable ./circuit_qaoa/qasm/${Q}/qaoa${Q}.qasm ${D}
done 

echo "Running HyQuas"
cd ~/FOR-QAOA-GPU/HyQuas/
cd ./third-party/cutt
make clean && make -j
mkdir -p evaluator-preprocess/parameter-files
cd ./benchmark && ./preprocess.sh

source ~/FOR-QAOA-GPU/HyQuas/scripts/init.sh -DBACKEND=mix -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=on -DMICRO_BENCH=on -DUSE_DOUBLE=on -DDISABLE_ASSERT=off -DENABLE_OVERLAP=on -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=on -DUSE_MPI=off -DMAT=7
for Q in {26..27};
do
    CUDA_VISIBLE_DEVICES=0 ./main ../../circuit_qaoa_hyquas/qasm/${Q}/qaoa${Q}.qasm
done 

echo "Running FOR-QAOA"
cd ~/FOR-QAOA-GPU/
make clean && make -j CHUNK_QUBIT=11
for Q in {26..27};
do
    B=$((Q-3))
    ./weighted ${P} ${Q} ${D} ${C} ${B}
done
