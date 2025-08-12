/*
 * Copyright 2025 The FOR-QAOA Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
#ifndef _STATE_H_
#define _STATE_H_

#include <nccl.h>
#define RANK_PER_NODE 8

typedef double Fp;

typedef unsigned long long ull;
typedef long long ll;

typedef struct  Complex{
    Fp real;
    Fp imag;
} Complex;

typedef struct {
    Complex* dState;
    Complex* dBuf;
    Complex* dTexMem;
    cudaTextureObject_t texObj;
    cudaStream_t compute_stream;
    ncclComm_t comm;
    void * stateHandle;
    void * bufHandle;
    int id;
#if USE_MPI
    int world_rank;
#endif
    Fp* d_graph_1D;
    int *d_subcirs;
    int *d_table;
} GPUState;


typedef struct State
{
    Complex* hState;
    int numDevice;
    int numQubits;
    int numQubitsPerDevice;
    int numBufsPerDevice;
    int numQubitsPerBuffer;
    ull numAmpsPerDevice;
    ull stateSizePerDevice;
    ull numAmpsPerBuf;
    ull bufSize;
    GPUState *gpus;
} State;

State createMPIState(int& N, int& D, int& B, int& C, int& world_rank, int& world_size);
State createState(int N, int D, int B, int C);
void destroyState(const State& state);

void stateCpyDeviceToHost(const State& state);
void stateCpyHostToDevice(const State& state);

void stateInitDeviceState(const State& qureg);
void ncclWarnup(State &qureg);

void prepare(State &qureg,int C);
#endif