#!/usr/bin/env python3
from qiskit import transpile, qasm2
import qiskit_aer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your script.')
    parser.add_argument('qasm2_path', type=str, help='Path to the qasm2 file')
    parser.add_argument('D', type=int, help='Exponent of 2, where 2^D denotes the number of devices')
    parser.add_argument('--fusion_enable', action='store_true', help='Enable gate fusion (see qiskit.AerSimulator for details)')
    parser.add_argument('--cuStateVec_enable', action='store_true', help='Enable cuQuatnum for backend')
    # parser.add_argument('--no_warnup', action='store_false', help='Number of blocking qubits  (see qiskit.AerSimulator for details)')
    
    args = parser.parse_args()
    qasm2_path = args.qasm2_path
    D = args.D
    fusion_enable = args.fusion_enable
    # is_warnup = args.warnup
    
    cuStateVec_enable = args.cuStateVec_enable
    
    simulator = qiskit_aer.AerSimulator(
        method='statevector', 
        device='GPU', 
        precision='double', 
        fusion_enable=fusion_enable, 
        # cache blocking options, we set blocking_qubits to 23 accroding to the following turotial
        # https://qiskit.github.io/qiskit-aer/howtos/running_gpu.html
        blocking_enable=True,
        blocking_qubits=23,
    )


    qc = qasm2.load(qasm2_path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    qc.measure_all()
    if fusion_enable:
        qc = transpile(qc, simulator, optimization_level=3)
    job = simulator.run(qc, cuStateVec_enable=cuStateVec_enable, target_gpus=list(range(0, 1<<D)))
    res = job.result()
    print(f"{res.metadata['time_taken_execute']:.4f}")
