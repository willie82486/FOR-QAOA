import os
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
outputfile = "output.log"
f = open(outputfile, 'w')
#====================================================================
# Single GPU
#====================================================================
#------------------------------------------------------
# QuEST
#------------------------------------------------------
# P = (1, 5)
# print("QuEST")
# for p in P:
#     out_str = f'./QuEST/QuEST/build/demo {p}'
#     os.system(out_str)
#     # print(out_str)

#------------------------------------------------------
# Observing the relation between Chunk size and qubits
#------------------------------------------------------
# P = [5]
# qubits = range(20, 32)
# # C = range(5, 12)
# C = [11]
# D = 0 # num of Device

# print("============Single GPU Weighted============")

# for c in C:  
#     os.system(f"make clean && make -j CHUNK_QUBIT={c}")  
#     for p in P:
#         for n in qubits:
#             B = n - 3
#             out_str = f'./weighted {p} {n} {D} {c} {B} >> {outputfile}'
#             os.system(out_str)

#------------------------------------------------------
# Observing the relation between Chunk size and Unrolling
#------------------------------------------------------
# U have to adjust the parameter of Unroll manually.
# P = [10]
# qubits = 31
# C = range(5, 12)
# D = 0 # num of Device

# for c in C:  
#     os.system(f"make clean && make -j CHUNK_QUBIT={c}")  
#     for p in P:
#             B = qubits - 3
#             out_str = f'./weighted {p} {qubits} {D} {c} {B} >> {outputfile}'
#             os.system(out_str)

#====================================================================
# Multi GPU
#====================================================================
#------------------------------------------------------
# weak scaling : running num of device form 0(with qubits : 31) ~  3 (with qubits : 34)
#------------------------------------------------------
# P = (1, 5)
# qubits = 31
# C = 11 # set the value which performs the best effect in single GPU
# D = (0, 1, 2, 3)
# print("============Multi GPU Weighted============")

# os.system(f"make clean && make -j CHUNK_QUBIT={C}")
# for p in P:
#     for d in D:
#         B = qubits - 3
#         if d == 3:
#             B = qubits - 4
#         out_str = f'./weighted {p} {qubits+d} {d} {C} {B} >> {outputfile}'
#         os.system(out_str)
            
#------------------------------------------------------            
# strong scaling : qubit fix on 31, and running num of device from 0 ~ 3  
#------------------------------------------------------
# P = (1, 5)
# qubits = 31
# C = 11 # set the value which perform the best effect in single GPU
# D = (0, 1, 2, 3)
# print("============Multi GPU Weighted============")

# os.system(f"make clean && make -j CHUNK_QUBIT={C}")
# for p in P:
#     for d in D:
#         B = qubits - 3
#         out_str = f'./weighted {p} {qubits} {d} {C} {B} >> {outputfile}'
#         os.system(out_str)


#------------------------------------------------------            
# Impact of transferred buffer size on 8 GPU  
#------------------------------------------------------
# P = (1, 5)
# qubits = 31
# C = 11 # set the value which perform the best effect in single GPU
# D = 3
# B = range(10, qubits-3)
# print("============Impact of transferred buffer size on 8 GPU ============")

# os.system(f"make clean && make -j CHUNK_QUBIT={C}")
# for p in P:
#     for b in B:
#         out_str = f'./weighted {p} {qubits} {D} {C} {b} >> {outputfile}'
#         os.system(out_str)
            
#------------------------------------------------------            
# Impact of Qubit size on 8 GPU  
#------------------------------------------------------
# P = (1, 5)
# qubits = range(20, 32)
# C = 11 # set the value which perform the best effect in single GPU
# D = 3
# print("============Impact of transferred buffer size on 8 GPU ============")

# os.system(f"make clean && make -j CHUNK_QUBIT={C}")
# for p in P:
#     for q in qubits:
#         B = q - 3
#         out_str = f'./weighted {p} {q} {D} {C} {B} >> {outputfile}'
#         os.system(out_str)

#------------------------------------------------------            
# Without NVLink
#------------------------------------------------------
# P = (1, 5)
# qubits = 31
# C = 11 # set the value which perform the best effect in single GPU
# D = 3
# B = range(10, qubits-3)
# print("============Impact of transferred buffer size on 8 GPU ============")

# os.system(f"make clean && make -j CHUNK_QUBIT={C}")
# for p in P:
#     for b in B:
#         out_str = f'NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 ./weighted {p} {qubits} {D} {C} {b} >> {outputfile}'
#         os.system(out_str)



#====================================================================
# ChunkTuning
#====================================================================
P = 1
C = range(5, 13)
# qubits = range(20, 30)
q = 30
os.system(f"cd ../../")
for chunk_size in C:
    os.system(f"make clean && make -j CHUNK_QUBIT={chunk_size}")
    print("chunk_size :", chunk_size)
    out_str = f'./weighted {P} {q} {chunk_size} >> {outputfile}'
    os.system(out_str)      
