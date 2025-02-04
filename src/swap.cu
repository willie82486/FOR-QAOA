#include <iostream>
#include <math.h>
#include "swap.h"
#include "state.h"
#include "helper_cuda.hpp"
#include <pthread.h>
#include <algorithm>
void CSQS(const State &qureg,  const int csqsSize);
void CSQS(const State &qureg,  const int csqsSize, char* targ);
void mapping_after_swap_conti(const State& qureg, Fp* graph,const int SwapOut,const int SwapIn,const int numSwap){
    assert(SwapIn > SwapOut);
    assert(SwapIn + numSwap -1 <= qureg.numQubits);
    int num = qureg.numQubits;
    int base_Out = SwapOut * num;
    int base_In = SwapIn * num;
    int interval = SwapIn - SwapOut - numSwap ;

    //targ 0 of RZZ belongs to SwapOut and targ1 belongs to SwapIn
    for(int i = 0; i< SwapOut; i++)
    {
        int OutIndex = i * num + SwapOut; 
        int InIndex = i * num + SwapIn;
        for(int j = 0; j < numSwap; j++ ){
            auto tmp = graph[OutIndex+j];
            graph[OutIndex+j]= graph[InIndex+j];
            graph[InIndex+j] = tmp;
        }
    }



    for(int j = SwapIn + numSwap ; j < num ; j++){
        for(int i = 0; i < numSwap; i++){
            auto tmp = graph[base_Out + i*num + j];
            graph[base_Out+ i*num +j]= graph[base_In+ i*num +j];
            graph[base_In+ i*num +j] = tmp;         
        }

    }

    
    // Both of targ0 and targ1 belongs to the same section(SwapOut or SwapIn)
    for(int i = 0;i < numSwap;i++)
    {
        int offset_swapout = SwapOut + i;
        int offset_swapin = SwapIn + i;
        while(offset_swapout < SwapOut + numSwap){
            auto tmp = graph[base_Out + i * num + offset_swapout];
            graph[base_Out + i * num + offset_swapout]= graph[base_In + i * num + offset_swapin];
            graph[base_In + i * num + offset_swapin] = tmp;
            offset_swapout++;
            offset_swapin++;
        }
    } 


    // std::cout<< "interval = "<<interval<<std::endl;
    int Out = base_Out + SwapOut + numSwap;
    int In = base_Out + numSwap * num + SwapIn;
    for(int i = 0; i < interval ; i++)
    {
        for(int j = 0; j < numSwap ; j++)
        {
            auto tmp = graph[Out + j*num + i];
            graph[Out + j*num + i] = graph[In + i*num + j];
            graph[In + i*num + j] = tmp;
        }
    }
    
    int start = base_Out + SwapIn ;
    for(int i = 1 ; i< numSwap ;i++){
        for(int j = 0;j < i ;j++){
            auto tmp = graph[start + i*num +j];
            graph[start + i*num +j] = graph[start + i + j*num];
            graph[start + i + j*num] = tmp;
        }
    }
    
}
void mapping_after_swap_conti(const State& qureg, ull* graph,const int SwapOut,const int SwapIn,const int numSwap){
    assert(SwapIn > SwapOut);
    assert(SwapIn + numSwap -1 <= qureg.numQubits);
    int num = qureg.numQubits;
    int base_Out = SwapOut * num;
    int base_In = SwapIn * num;
    int interval = SwapIn - SwapOut - numSwap ;
 
    //targ 0 of RZZ belongs to SwapOut and targ1 belongs to SwapIn
    for(int i = 0; i< SwapOut; i++)
    {
        int OutIndex = i * num + SwapOut; 
        int InIndex = i * num + SwapIn;
        for(int j = 0; j < numSwap; j++ ){
            auto tmp = graph[OutIndex+j];
            graph[OutIndex+j]= graph[InIndex+j];
            graph[InIndex+j] = tmp;
        }
    }



    for(int j = SwapIn + numSwap ; j < num ; j++){
        for(int i = 0; i < numSwap; i++){
            auto tmp = graph[base_Out + i*num + j];
            graph[base_Out+ i*num +j]= graph[base_In+ i*num +j];
            graph[base_In+ i*num +j] = tmp;         
        }

    }

    
    // Both of targ0 and targ1 belongs to the same section(SwapOut or SwapIn)
    for(int i = 0;i < numSwap;i++)
    {
        int offset_swapout = SwapOut + i;
        int offset_swapin = SwapIn + i;
        while(offset_swapout < SwapOut + numSwap){
            auto tmp = graph[base_Out + i * num + offset_swapout];
            graph[base_Out + i * num + offset_swapout]= graph[base_In + i * num + offset_swapin];
            graph[base_In + i * num + offset_swapin] = tmp;
            offset_swapout++;
            offset_swapin++;
        }
    } 


    // std::cout<< "interval = "<<interval<<std::endl;
    int Out = base_Out + SwapOut + numSwap;
    int In = base_Out + numSwap * num + SwapIn;
    for(int i = 0; i < interval ; i++)
    {
        for(int j = 0; j < numSwap ; j++)
        {
            auto tmp = graph[Out + j*num + i];
            graph[Out + j*num + i] = graph[In + i*num + j];
            graph[In + i*num + j] = tmp;
        }
    }
    
    int start = base_Out + SwapIn ;
    for(int i = 1 ; i< numSwap ;i++){
        for(int j = 0;j < i ;j++){
            auto tmp = graph[start + i*num +j];
            graph[start + i*num +j] = graph[start + i + j*num];
            graph[start + i + j*num] = tmp;
        }
    }
    
}

__device__ __forceinline__
ull bit_string(const ull task, const int targ0, const int targ1)
{
	ull res = task;
	ull mask;
	mask = (1ULL << targ0) - 1;
	res = ((res >> targ0) << (targ0+1)) | (res & mask);
	mask = (1ULL << targ1) - 1;
	res = ((res >> targ1) << (targ1+1)) | (res & mask);
    return res;
}

__device__ __forceinline__
long long bit_string_multi(const long long task, const int targ, const int amount)
{
	long long res = task;
	long long mask;
	mask = (1ULL << targ) - 1;
	res = ((res >> targ) << (targ+amount)) | (res & mask);
    return res;
}

__global__ 
void _swap_gate(Complex *sv,const int targ0,const int targ1) {//targ0 must less than targ1 

    ull tidx = blockIdx.x * blockDim.x + threadIdx.x;
    
	ull idx = bit_string(tidx, targ0, targ1);

    ull off01 = idx + (1ULL<<targ0);
    ull off10 = idx + (1ULL<<targ1);
    Complex q01;
    Complex q10;

	q01 = sv[off01];
	q10 = sv[off10];
	sv[off01] = q10;
	sv[off10] = q01;
}

void swap_gate(const State& qureg,const int targ0,const int targ1){

        ull grid = 1;
        ull block = qureg.numAmpsPerDevice >> 2;

        if (block > 512) {
            grid = block / 512;
            block = 512;
        }

        for (int dev = 0; dev < qureg.numDevice; dev++) {
            checkCudaErrors(cudaSetDevice(dev));
            _swap_gate<<<grid, block>>>(qureg.gpus[dev].dState, targ0, targ1);
        }
        for (int dev = 0; dev < qureg.numDevice; dev++) {
            checkCudaErrors(cudaSetDevice(dev));
            checkCudaErrors(cudaDeviceSynchronize());
        }

        return;
    
}

__global__
void _swap_gate_conti(Complex* sv, const  int swapOut, const int swapIn, const int numSwap, const int* d_table){
    ull tidx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int numSwapBits = (numSwap<<1)-1; 

    size_t gidx = tidx >> numSwapBits;

    gidx = bit_string_multi(gidx, swapOut, numSwap);//Swapout < Swapin
    gidx = bit_string_multi(gidx, swapIn, numSwap);


    int swapIdx = tidx & ((1<<numSwapBits)-1);

    if (swapIdx >= ((1<<numSwap) * ((1<<numSwap)-1) / 2))
    return;

    size_t s1 = d_table[2*swapIdx];
    size_t s2 = d_table[2*swapIdx+1];

    size_t up = (s1 << swapIn) + (s2 << swapOut) + gidx;
    size_t lo = (s1 << swapOut) + (s2 << swapIn) + gidx;

    Complex tmp = sv[up];
    sv[up] = sv[lo];
    sv[lo] = tmp;
}

void swap_gate_conti(const State& qureg, const int SwapOut, const int SwapIn, int numSwap){
        assert(SwapIn > SwapOut);
        if(qureg.numDevice > 1 && SwapIn + numSwap -1 >= qureg.numQubitsPerDevice){
            int numQubitOutofDevice = SwapIn + numSwap - qureg.numQubitsPerDevice; //number of qubit out of device
            if(SwapIn < qureg.numQubitsPerDevice){
                swap_gate_conti(qureg,  
                                qureg.numQubitsPerDevice - SwapIn, 
                                qureg.numQubitsPerDevice - numQubitOutofDevice, 
                                numQubitOutofDevice);
                char* targ = (char*)malloc((SwapIn + numSwap - qureg.numQubitsPerDevice) * sizeof(char));
                for(int i = 0;i< SwapIn + numSwap - qureg.numQubitsPerDevice ;i++){
                    targ[i] = i;
                }
                CSQS(qureg, SwapIn + numSwap - qureg.numQubitsPerDevice, targ );
                swap_gate_conti(qureg, 
                                qureg.numQubitsPerDevice - SwapIn, 
                                qureg.numQubitsPerDevice - numQubitOutofDevice, 
                                numQubitOutofDevice);
            }
            else{
                swap_gate_conti(qureg, 0, qureg.numQubitsPerDevice - numSwap, numSwap);
                char* targ = (char*)malloc(numSwap * sizeof(char));
                for(int i = 0;i< numSwap ;i++){
                    targ[i] = SwapIn + i - qureg.numQubitsPerDevice;
                }
                CSQS(qureg, numSwap, targ);
                swap_gate_conti(qureg, 0, qureg.numQubitsPerDevice - numSwap, numSwap);
                return;
            }
            numSwap -= numQubitOutofDevice; 
            if(!numSwap)return;;
        }
        assert(SwapIn + numSwap -1 <= qureg.numQubitsPerDevice);
        ull grid = 1;
        ull block = qureg.numAmpsPerDevice >> 1;
        if (block > 512) {
            grid = block / 512 ;
            block = 512;
        }
		ull numSwapPairs = (1ull<<numSwap) * ((1ull<<numSwap)-1) / 2;
        ull swap_table_size = numSwapPairs * 2 * sizeof(int);
		int *table;

		table = (int*) malloc(swap_table_size);

		int counter = 0;
		for (int i=0; i<(1<<numSwap); i++)
			for (int j=i+1; j<(1<<numSwap); j++, counter++) {
				table[2*counter] = i;
				table[2*counter+1] = j;
			}

		assert(counter == numSwapPairs);

#if USE_MPI
        checkCudaErrors(cudaMemcpyAsync(qureg.gpus->d_table, table, swap_table_size, cudaMemcpyHostToDevice, qureg.gpus->compute_stream));
        // checkCudaErrors(cudaMemcpy(qureg.gpus->d_table, table, swap_table_size, cudaMemcpyHostToDevice));
        _swap_gate_conti<<<grid, block, 0 ,qureg.gpus->compute_stream>>>(qureg.gpus->dState, SwapOut, SwapIn, numSwap, qureg.gpus->d_table);

#else
        for (int dev = 0; dev < qureg.numDevice; dev++) {
            checkCudaErrors(cudaSetDevice(dev));
            checkCudaErrors(cudaMemcpyAsync(qureg.gpus[dev].d_table, table, swap_table_size, cudaMemcpyHostToDevice, qureg.gpus[dev].compute_stream));
            _swap_gate_conti<<<grid, block, 0 ,qureg.gpus[dev].compute_stream>>>(qureg.gpus[dev].dState, SwapOut, SwapIn, numSwap, qureg.gpus[dev].d_table);
            // checkCudaErrors(cudaFreeAsync(qureg.gpus[dev].d_table,qureg.gpus[dev].compute_stream));
        }
#endif

}

__host__ __device__ __forceinline__
uint64_t insert_bit_0(uint64_t task, const char targ)
{
	uint64_t mask = (1ULL << targ) - 1;
	return ((task >> targ) << (targ+1)) | (task & mask);
}


template<int N>
__host__ __device__ __forceinline__
uint64_t insert_bits_0(uint64_t task, const char targs[])
{
	#pragma unroll
	for (int i = 0; i < N; i++)
		task = insert_bit_0(task, targs[i]);
	return task;
}


void CSQS(const State &qureg,  const int csqsSize, char* targ)
{
    const int &N = qureg.numQubits;
    const int &D = qureg.numQubits - qureg.numQubitsPerDevice;
    const int &B = qureg.numQubitsPerBuffer;

    const int numGroups = 1 << (D - csqsSize);
    const int numMemberInGroup = 1 << csqsSize;

    std::vector<std::vector<int>> devlist(numGroups, std::vector<int>(numMemberInGroup));
    // Deal with all to all problem.
    for (int mbr = 0; mbr < numMemberInGroup; mbr++) {
        int mem_bits = 0;
        for (int i = 0; i < csqsSize; i++)
            mem_bits |= ((mbr >> i) & 1) << targ[i];
        devlist[0][mbr] = mem_bits;
    }
   
    for (int grp = 1; grp < (numGroups); grp++)
    {
        int grp_bits = grp;
        for (int i = 0; i < (D - csqsSize); i++)
            grp_bits = insert_bits_0<1>(grp_bits, &targ[i]);
        for (int mbr = 0; mbr < numMemberInGroup; mbr++)
            devlist[grp][mbr] = grp_bits | devlist[0][mbr];
    }

 
    // Using buffer to tackle ncclSend and ncclRecv
    for (ull off = 0; off < (1ull << (N - D - csqsSize)); off += (1 << (B - csqsSize)))
    {
        for (int grp = 0; grp < (1 << (D - csqsSize)); grp++)
        {
            checkCudaErrors(ncclGroupStart());
            for (int a = 0; a < (1 << csqsSize); a++)
            {
                int devA = devlist[grp][a];
#if USE_MPI
                if(qureg.gpus->world_rank == devA){
                    for (int b = 0; b < (1 << csqsSize); b++)
                    {
                        if (a == b)
                            continue;
                        int devB = devlist[grp][b];
                        int off_sv = b * (1ull << (N - D - csqsSize)) + off;
                        int off_bf = b * (1ull << (B - csqsSize));

                        checkCudaErrors(ncclSend(qureg.gpus->dState + off_sv, (1 << (B - csqsSize)) * sizeof(Complex), ncclChar, devB, qureg.gpus->comm, qureg.gpus->compute_stream)); 
                        checkCudaErrors(ncclRecv(qureg.gpus->dBuf + off_bf, (1 << (B - csqsSize)) * sizeof(Complex), ncclChar, devB, qureg.gpus->comm, qureg.gpus->compute_stream));
                    }
                }

#else
                // checkCudaErrors(ncclGroupStart());
                for (int b = 0; b < (1 << csqsSize); b++)
                {
                    if (a == b)
                        continue;
                    int devB = devlist[grp][b];
                    int off_sv = b * (1ull << (N - D - csqsSize)) + off;
                    int off_bf = b * (1ull << (B - csqsSize));

                    checkCudaErrors(ncclSend(qureg.gpus[devA].dState + off_sv, (1 << (B - csqsSize)) * sizeof(Complex), ncclChar, devB, qureg.gpus[devA].comm, qureg.gpus[devA].compute_stream)); 
                    checkCudaErrors(ncclRecv(qureg.gpus[devA].dBuf + off_bf, (1 << (B - csqsSize)) * sizeof(Complex), ncclChar, devB, qureg.gpus[devA].comm, qureg.gpus[devA].compute_stream));
                }

#endif
                // checkCudaErrors(ncclGroupEnd());
            }
            checkCudaErrors(ncclGroupEnd());
        }

        for (int grp = 0; grp < (1 << (D - csqsSize)); grp++)
        {
            // checkCudaErrors(ncclGroupStart());
            for (int a = 0; a < (1 << csqsSize); a++)
            {
                int devA = devlist[grp][a];
#if USE_MPI
                if(qureg.gpus->world_rank == devA){
                    for (int b = 0; b < (1 << csqsSize); b++)
                    {
                        if (a == b)
                            continue;
                        int off_sv = b * (1ull << (N - D - csqsSize)) + off;
                        int off_bf = b * (1ull << (B - csqsSize));
                        checkCudaErrors(cudaMemcpyAsync(qureg.gpus->dState + off_sv, qureg.gpus->dBuf + off_bf, (1 << (B - csqsSize)) * sizeof(Complex), cudaMemcpyDeviceToDevice, qureg.gpus->compute_stream));
                        // checkCudaErrors(cudaMemcpy(qureg.gpus->dState + off_sv, qureg.gpus->dBuf + off_bf, (1 << (B - csqsSize)) * sizeof(Complex), cudaMemcpyDeviceToDevice));
                    }
                }

#else
                // checkCudaErrors(ncclGroupStart());
                for (int b = 0; b < (1 << csqsSize); b++)
                {
                    if (a == b)
                        continue;
                    int off_sv = b * (1ull << (N - D - csqsSize)) + off;
                    int off_bf = b * (1ull << (B - csqsSize));
                    checkCudaErrors(cudaMemcpyAsync(qureg.gpus[devA].dState + off_sv, qureg.gpus[devA].dBuf + off_bf, (1 << (B - csqsSize)) * sizeof(Complex), cudaMemcpyDeviceToDevice, qureg.gpus[devA].compute_stream));
                }
#endif
                // checkCudaErrors(ncclGroupEnd());
            }
            // checkCudaErrors(ncclGroupEnd());
        }
    }

}