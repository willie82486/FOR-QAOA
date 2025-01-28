#!/usr/bin/env python3

# pip3 install qiskit
# pip3 install qiskit-aer-gpu

import os
import numpy as np
import qiskit_aer
from qiskit import QuantumCircuit
import math
# 請調整這邊的參數
N = 30
gamma =  math.pi / 4
beta = math.pi / 4
# QAOA sim dump 出來的 stetevector
dmp_path = "dmp.txt"
C =10

D = 0
simulator = qiskit_aer.AerSimulator(
    method='statevector', 
    device='GPU', 
    precision='double', 
    fusion_enable=False,
    blocking_enable=False,
)

def statevector(qc):
    qc.save_statevector()
    job = simulator.run(qc, target_gpus=list(range(0, 1<<D)))
    return job.result().data()['statevector'].data

def validate(sv):
    return np.all(np.abs(sv-file_state) < 1e-9)

qc = QuantumCircuit(N, N)

for i in range(N):
    qc.h(i)
    
for i in range( N-1):
    for j in range(i+1, N):
        qc.rzz(gamma, i, j)

for i in range(N):
    qc.rx(beta, i)

    
# for i in range(10):
#     qc.rx(beta,i)
# for i in range(9):
#     qc.swap(0+i,10+i)
# for i in range(10):
#     qc.rx(beta,i)
    
# qc.rx(beta, 0)





# print("single gpu")
# os.system(f"./weighted 1 {N} 0 10 8")
# ans_state = statevector(qc)
# file_state = np.fromfile(dmp_path, dtype=np.complex128)

# print(ans_state[:64])
# print(file_state[:64])

# print(np.max(np.abs(ans_state-file_state)))

# print(np.all(np.abs(ans_state-file_state) < 1e-8))


os.system(f"make clean && make -j CHUNK_QUBIT={C}")
print("multigpu:")
os.system("rm dmp.txt")
os.system(f"./weighted 1 {N} 0 {C} 27")
ans_state = statevector(qc)
file_state = np.fromfile(dmp_path, dtype=np.complex128)

print(ans_state[:16])
print(file_state[:16])

print(np.max(np.abs(ans_state-file_state)))

print(np.all(np.abs(ans_state-file_state) < 1e-8))


