#include "rzz.h"

#include <math.h>

#include "state.h"
#include "helper_cuda.h"

Fp initialAmp;

void initH(const State& qureg) {
    initialAmp = pow(1./sqrt(2.), qureg.numQubits);
}

// __global__
// void _rotationCompressionUnweighted(Complex* sv, int numQubits, ull b_offset, Fp angle, ull* graph, int numGates, bool isFirstLayer) {
//     ull tidx = blockIdx.x * blockDim.x + threadIdx.x;
//     ull b = tidx + b_offset;
    
//     int Cp = 0;
//     // count number of 1
//     for (int i = 0; i < numQubits; i++) {
//         ull strxor;
//         ull msk = -((b >> i) & 1LL);
//         strxor = graph[i] & (b ^ msk);
//         Cp += __popcll(strxor);
//     }

//     Complex contracted_rzz;
//     sincosf((2*Cp-numGates)*angle/2, &contracted_rzz.imag, &contracted_rzz.real);

//     if (isFirstLayer) {
//         sv[tidx].real = contracted_rzz.real * initialAmp;
// 	    sv[tidx].imag = contracted_rzz.imag * initialAmp;
//     } else {
//         Complex in = sv[tidx];
//         sv[tidx].real = in.real * contracted_rzz.real - in.imag * contracted_rzz.imag;
//         sv[tidx].imag = in.real * contracted_rzz.imag + in.imag * contracted_rzz.real;
//     }
// }

// void rotationCompressionUnweighted(const State& qureg, Fp angle, ull* graph, bool isFirstLayer) {
//     int numGates = 0;
//     for (int i = 0; i < qureg.numQubits; i++) {
//         numGates += __builtin_popcountll(graph[i]);
//     }

//     ull grid = 1;
//     ull block = qureg.numAmpsPerDevice;

//     if (block > 128) {
//         grid = block / 128;
//         block = 128;
//     }

//     for (int dev = 0; dev < qureg.numDevice; dev++) {
//         ull b_offset = dev * qureg.numAmpsPerDevice;

//         checkCudaErrors(cudaSetDevice(dev));
//         _rotationCompressionUnweighted<<<grid, block>>>(qureg.gpus[dev].dState, qureg.numQubits, b_offset, angle, graph, numGates, isFirstLayer);
//     }

//     for (int dev = 0; dev < qureg.numDevice; dev++) {
//         checkCudaErrors(cudaSetDevice(dev));
//         checkCudaErrors(cudaDeviceSynchronize());
//     }
// }

__global__
void _rotationCompressionWeighted(Complex* sv,const int numQubits,const int b_offset,const Fp angle, const __restrict__ Fp* graph,const bool isFirstLayer, Fp iniAmp_gpu) {
    //old version
    ull tidx = blockIdx.x * blockDim.x + threadIdx.x;
    ull b = tidx + b_offset;

    Fp sum = 0;
    // count number of 1
    for (int i = 0; i < numQubits; i++) {
        for (int j = i + 1; j < numQubits; j++) {
            int idx = i*numQubits+j;
            if (graph[idx] != 0) {
                if (((b>>i)&0b1) ^ ((b>>j)&0b1)) {
                    sum += graph[idx];
                } else {
                    sum -= graph[idx];
                }
            }
        }
    }


    Complex contracted_rzz;
    sincos(sum*angle/2, &contracted_rzz.imag, &contracted_rzz.real);

    if (isFirstLayer) {
        sv[tidx].real = contracted_rzz.real * iniAmp_gpu;
	    sv[tidx].imag = contracted_rzz.imag * iniAmp_gpu;
    } else {
        Complex in = sv[tidx];
        sv[tidx].real = in.real * contracted_rzz.real - in.imag * contracted_rzz.imag;
        sv[tidx].imag = in.real * contracted_rzz.imag + in.imag * contracted_rzz.real;
    }
}



__global__
void _rotationCompressionWeighted_conti(Complex* sv,const int numQubits,const int b_offset,const Fp angle, const __restrict__ Fp* graph,const bool isFirstLayer, Fp iniAmp_gpu) {
    //new version (need to transform the graph into 1D array first)
    ull tidx = blockIdx.x * blockDim.x + threadIdx.x;
    ull b = tidx + b_offset;

    Fp sum = 0;
    // count number of 1
    // int rec = numQubits - 1, j = 1;
    int cnt = 0;
    for (int i = 0; i < numQubits; i++) {
        for (int j = i + 1; j < numQubits; j++) {
            // int idx = i*(numQubits-i)+j;
            if (graph[cnt] != 0) {
                if (((b>>i)&0b1) ^ ((b>>j)&0b1)) {
                    sum += graph[cnt];
                } else {
                    sum -= graph[cnt];
                }
            }
            cnt++;
        }
    }

    Complex contracted_rzz;
    sincos(sum*angle/2, &contracted_rzz.imag, &contracted_rzz.real);

    if (isFirstLayer) {
        sv[tidx].real = contracted_rzz.real * iniAmp_gpu;
	    sv[tidx].imag = contracted_rzz.imag * iniAmp_gpu;
    } else {
        Complex in = sv[tidx];
        sv[tidx].real = in.real * contracted_rzz.real - in.imag * contracted_rzz.imag;
        sv[tidx].imag = in.real * contracted_rzz.imag + in.imag * contracted_rzz.real;
    }
}

Fp* UpperTri_to_1D(const int N, const __restrict__ Fp* uppertri) {

    size_t size = (N * (N-1))/2 * sizeof(Fp);
    Fp* graph = (Fp*)malloc(size);
    int cnt = 0;
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            graph[cnt] = uppertri[i*N + j];
            cnt++;
        }
    }

    return graph;
}

void rotationCompressionWeighted(State& qureg,const Fp angle,const __restrict__ Fp* graph,const bool isFirstLayer) {
    ull grid = 1;
    ull block = qureg.numAmpsPerDevice;

    if (block > 128) {
        grid = block / 128;
        block = 128;
    }
    Fp* graph_1d = UpperTri_to_1D(qureg.numQubits, graph);
    // Fp* d_graph_1D;
    size_t size = (qureg.numQubits * (qureg.numQubits-1))/2 * sizeof(Fp);

#if USE_MPI
    ull b_offset = qureg.gpus->id * qureg.numAmpsPerDevice;
    checkCudaErrors(cudaMemcpyAsync(qureg.gpus->d_graph_1D, graph_1d, size, cudaMemcpyHostToDevice, qureg.gpus->compute_stream));
    _rotationCompressionWeighted_conti<<<grid, block, 0, qureg.gpus->compute_stream>>>(qureg.gpus->dState, qureg.numQubits, b_offset, angle, qureg.gpus->d_graph_1D, isFirstLayer, initialAmp);
#else
    for (int dev = 0; dev < qureg.numDevice; dev++) {
        ull b_offset = dev * qureg.numAmpsPerDevice;

        checkCudaErrors(cudaSetDevice(dev));
        // checkCudaErrors(cudaMallocAsync(&qureg.gpus[dev].d_graph_1D, size, qureg.gpus[dev].compute_stream));
        // checkCudaErrors(cudaMemcpyAsync(d_graph_1D, graph_1d, size, cudaMemcpyHostToDevice, qureg.gpus[dev].compute_stream));
        checkCudaErrors(cudaMemcpyAsync(qureg.gpus[dev].d_graph_1D, graph_1d, size, cudaMemcpyHostToDevice, qureg.gpus[dev].compute_stream));
        // _rotationCompressionWeighted<<<grid, block, 0, qureg.gpus[dev].compute_stream>>>(qureg.gpus[dev].dState, qureg.numQubits, b_offset, angle, graph, isFirstLayer, initialAmp);
        _rotationCompressionWeighted_conti<<<grid, block, 0, qureg.gpus[dev].compute_stream>>>(qureg.gpus[dev].dState, qureg.numQubits, b_offset, angle, qureg.gpus[dev].d_graph_1D, isFirstLayer, initialAmp);
    // checkCudaErrors(cudaFreeAsync(d_graph_1D, qureg.gpus[dev].compute_stream));
    }
#endif
    // for (int dev = 0; dev < qureg.numDevice; dev++) {
    //     checkCudaErrors(cudaSetDevice(dev));
    //     checkCudaErrors(cudaDeviceSynchronize());
    // }
}