#include "rx.h"
#include <iostream>
#include <math.h>

#include "state.h"
#include "helper_cuda.h"

__device__ __forceinline__
ull bit_string(const ull task, const int target) {
    ull mask = (1ull << target) - 1;
    return ((task >> target) << (target + 1)) | (task & mask);
}



template<int unrolls, int threads>
__inline__ __device__ void rx_gate_unroll(Complex* __restrict__ sv, const int target,const Fp cosTheta_2,const Fp sinTheta_2)
{
	const int close = 1;
	#pragma unroll
	for (int u=0; u < unrolls >> close; u++) {
		int tidx = threadIdx.x + u * threads;
		long long idx = bit_string(tidx, target);
		long long up_off = idx;
		long long lo_off = idx + (1ull << target);
		Complex q0 = sv[up_off];
		Complex q1 = sv[lo_off];
		Complex q0Out;
		Complex q1Out;
		q0Out.real = q0.real * cosTheta_2 + q1.imag * sinTheta_2;
		q0Out.imag = q0.imag * cosTheta_2 - q1.real * sinTheta_2;
		q1Out.real = q0.imag * sinTheta_2 + q1.real * cosTheta_2;
		q1Out.imag =-q0.real * sinTheta_2 + q1.imag * cosTheta_2;
		sv[up_off] = q0Out;
		sv[lo_off] = q1Out;
	}
}


template<int unrolls, int threads>
__global__ void _multirotateX_unroll(Complex * __restrict__ sv,const Fp cosTheta_2,const Fp sinTheta_2,const  int* targetQubits, const int gate_size)
{
	// __shared__ complexArray<1024> sharedBuffer;
    __shared__ Complex sharedBuffer[unrolls*threads];
	const long long gidx = blockIdx.x * (unrolls*threads);

	#pragma unroll
	for (int u=0; u<unrolls; u++) {
		int tidx = threadIdx.x + u * threads;
		Complex q = sv[gidx + tidx];
		sharedBuffer[tidx].real = q.real;
		sharedBuffer[tidx].imag = q.imag;
		// sharedBuffer.reals[tidx] = q.real;
		// sharedBuffer.imags[tidx] = q.imag;
	}
	__syncthreads();

	for(int i = 0; i < gate_size; i++) {
        rx_gate_unroll<unrolls, threads>(sharedBuffer, targetQubits[i], cosTheta_2, sinTheta_2);
		__syncthreads();
	}

	#pragma unroll
	for (int u=0; u<unrolls; u++) {
		int tidx = threadIdx.x + u * threads;
		Complex q;
		q.real = sharedBuffer[tidx].real;
		q.imag = sharedBuffer[tidx].imag;
		// q.real = sharedBuffer.reals[tidx];
		// q.imag = sharedBuffer.imags[tidx];
		sv[gidx + tidx] = q;
	}
    // cudafree(sharedBuffer);
}



__inline__ __device__ void rx_gate(Complex* sv,const int target,const Fp cosTheta_2,const Fp sinTheta_2)
{
    ull tidx = blockIdx.x * blockDim.x + threadIdx.x;
    ull up_off = bit_string(tidx, target);
    ull lo_off = up_off + (1ull << target);

    Complex up = sv[up_off];
    Complex lo = sv[lo_off];

    // double coeff = 1/sqrt(2);
    // sv[up_off].real = coeff * (up.real + lo.real);
	// sv[up_off].imag = coeff * (up.imag + lo.imag);
	// sv[lo_off].real = coeff * (up.real - lo.real);
	// sv[lo_off].imag = coeff * (up.imag - lo.imag);


    sv[up_off].real =   up.real * cosTheta_2 + lo.imag * sinTheta_2;
    sv[up_off].imag =   up.imag * cosTheta_2 - lo.real * sinTheta_2;
    sv[lo_off].real =   up.imag * sinTheta_2 + lo.real * cosTheta_2;
    sv[lo_off].imag = - up.real * sinTheta_2 + lo.imag * cosTheta_2;
}

__global__
void _multirotateX(Complex* sv,const Fp cosTheta_2,const Fp sinTheta_2,const  int* targetQubits,const int gate_size)
{	
	for(int i = 0; i < gate_size; i++) {
		rx_gate(sv, targetQubits[i], cosTheta_2, sinTheta_2);
		__syncthreads();
	}
}


void multiRotateX(State& qureg, vector<int> &targetQubits,const int gate_size,const Fp angle) 
{
    Fp sinTheta_2, cosTheta_2;
    sincos(angle/2, &sinTheta_2, &cosTheta_2);

    // target qubit is not the most significant bit
    // if (targetQubit < qureg.numQubitsPerDevice)
    if (targetQubits[gate_size-1] < qureg.numQubitsPerDevice){
        /*
        ull grid = 1;
        ull block = qureg.numAmpsPerDevice >> 1;

        if (block > 1024) {                  
            grid = block / 1024;
            block = 1024;
        }
        */
        // cout<<"grid = "<< grid << "block = " << block << endl;

        const int chunk_size = 1 << CHUNK_QUBIT;
        const int grid = qureg.numAmpsPerDevice / chunk_size;
        const int unroll = 8;
        const int thread = chunk_size / unroll;
#if USE_MPI
        checkCudaErrors(cudaMemcpyAsync(qureg.gpus->d_subcirs, 
                                        targetQubits.data(),
                                        gate_size * sizeof(int), 
                                        cudaMemcpyHostToDevice,
                                        qureg.gpus->compute_stream));
        // checkCudaErrors(cudaMemcpy(qureg.gpus->d_subcirs, 
        //                                 targetQubits.data(),
        //                                 gate_size * sizeof(int), 
        //                                 cudaMemcpyHostToDevice)); 
        _multirotateX_unroll<unroll, thread><<<grid, thread, 0, qureg.gpus->compute_stream>>>(qureg.gpus->dState, cosTheta_2, sinTheta_2, qureg.gpus->d_subcirs, gate_size);

#else
        for (int dev = 0; dev < qureg.numDevice; dev++) {
            checkCudaErrors(cudaSetDevice(dev));
            checkCudaErrors(cudaMemcpyAsync(qureg.gpus[dev].d_subcirs, 
                                            targetQubits.data(),
                                            gate_size * sizeof(int), 
                                            cudaMemcpyHostToDevice,
                                            qureg.gpus[dev].compute_stream));
            // _multirotateX<<<grid, block, 0, qureg.gpus[dev].compute_stream>>>(qureg.gpus[dev].dState, cosTheta_2, sinTheta_2, qureg.gpus[dev].d_subcirs, gate_size);
            _multirotateX_unroll<unroll, thread><<<grid, thread, 0, qureg.gpus[dev].compute_stream>>>(qureg.gpus[dev].dState, cosTheta_2, sinTheta_2, qureg.gpus[dev].d_subcirs, gate_size);
        }
#endif
        return;
    }
    std::cout << "Jump out of if" << std::endl;
}

__global__
void _rotateX(Complex* sv,const Fp cosTheta_2,const Fp sinTheta_2,const int target) 
{
    ull tidx = blockIdx.x * blockDim.x + threadIdx.x;
    ull up_off = bit_string(tidx, target);
    ull lo_off = up_off + (1ull << target);

    Complex up = sv[up_off];
    Complex lo = sv[lo_off];

    sv[up_off].real =   up.real * cosTheta_2 + lo.imag * sinTheta_2;
    sv[up_off].imag =   up.imag * cosTheta_2 - lo.real * sinTheta_2;
    sv[lo_off].real =   up.imag * sinTheta_2 + lo.real * cosTheta_2;
    sv[lo_off].imag = - up.real * sinTheta_2 + lo.imag * cosTheta_2;
}


void rotateX(const State& qureg,const int targetQubit,const Fp angle) 
{
    float sinTheta_2, cosTheta_2;
    sincos(angle/2, &sinTheta_2, &cosTheta_2);

    // target qubit is not the most significant bit
    if (targetQubit < qureg.numQubitsPerDevice) {
        ull grid = 1;
        ull block = qureg.numAmpsPerDevice >> 1;

        if (block > 128) {
            grid = block / 128;
            block = 128;
        }

        for (int dev = 0; dev < qureg.numDevice; dev++) {
            checkCudaErrors(cudaSetDevice(dev));
            _rotateX<<<grid, block>>>(qureg.gpus[dev].dState, cosTheta_2, sinTheta_2, targetQubit);

        }

        for (int dev = 0; dev < qureg.numDevice; dev++) {
            checkCudaErrors(cudaSetDevice(dev));
            checkCudaErrors(cudaDeviceSynchronize());
        }

        return;
    }

}
