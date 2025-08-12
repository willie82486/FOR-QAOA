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
 
#include "sqsGate.h"
#include <omp.h>

size_t calc_idxbits(size_t n, size_t mask)
{
    size_t idx = 0;
    while (n)
    {
        size_t bit = mask & -mask;
        mask ^= bit;
        idx |= bit * (n & 1);
        n >>= 1;
    }
    return idx;
}

size_t inc_idx(size_t n, size_t mask)
{
    return ((n | ~mask) + 1) & mask;
}

size_t dec_idx(size_t n, size_t mask)
{
    return (n - 1) & mask;
}

size_t sub_idx(size_t a, size_t b, size_t mask)
{
    return (a - b) & mask;
}

void do_swap_block(Complex* buffer, sqs_info_t &info,
					size_t idx1_base, size_t idx2_base,
					bool A_eq_B)
{
    size_t t_idx = 0;
    size_t a1_idx = 0, a2_idx = 0;
    size_t b1_idx, b2_idx;
    size_t idx1, idx2;
    do
    {
        do
        {
            if (A_eq_B && a2_idx == info.sub_b_mask)
                break;
            b1_idx = A_eq_B ? inc_idx(a2_idx, info.sub_b_mask) : 0;
            b2_idx = A_eq_B ? inc_idx(a1_idx, info.sub_a_mask) : 0;
            do
            {
                /* Calculate final index */
                idx1 = idx1_base | t_idx | a1_idx | b1_idx;
                idx2 = idx2_base | t_idx | a2_idx | b2_idx;

                /* Do Swapping */
                // std::cout << "idx1: " << std::bitset<QUBIT>(idx1);
                // std::cout << ", idx2: " << std::bitset<QUBIT>(idx2) << std::endl;
                Complex tmp = buffer[idx2];
                buffer[idx2] = buffer[idx1];
                buffer[idx1] = tmp;

                b1_idx = inc_idx(b1_idx, info.sub_b_mask);
                b2_idx = inc_idx(b2_idx, info.sub_a_mask);
            } while (b1_idx);

            a1_idx = inc_idx(a1_idx, info.sub_a_mask);
            a2_idx = inc_idx(a2_idx, info.sub_b_mask);
        } while (a1_idx);

        t_idx = inc_idx(t_idx, info.sub_task_mask);
    } while (t_idx);
}

void do_swap_row_subtask(Complex* buffer, sqs_info_t &info,
                            size_t T_idx, size_t A1_idx, size_t A2_idx)
{
    /* A == B */
    size_t B1_idx = A2_idx;
    size_t B2_idx = A1_idx;
    do_swap_block(buffer, info, T_idx | A1_idx | B1_idx,
                  T_idx | A2_idx | B2_idx, true);

    /* A < B */
    B1_idx = inc_idx(B1_idx, info.b_mask);
    B2_idx = inc_idx(B2_idx, info.a_mask);
    while (B1_idx)
    {
        do_swap_block(buffer, info,
                      T_idx | A1_idx | B1_idx,
                      T_idx | A2_idx | B2_idx, false);
        B1_idx = inc_idx(B1_idx, info.b_mask);
        B2_idx = inc_idx(B2_idx, info.a_mask);
    }
}

void do_swap_task_range(Complex* buffer, sqs_info_t &info,
                            size_t begin, size_t end)
{
    size_t T_begin = begin >> (info.sw_bits - 1);
    size_t T_end = end >> (info.sw_bits - 1);
    size_t M_begin = begin & ((1ULL << (info.sw_bits - 1)) - 1);
    size_t M_end = end & ((1ULL << (info.sw_bits - 1)) - 1);
    // size_t M_row_end = 1ULL << (info.sw_bits - 1); 

    size_t T_idx = calc_idxbits(T_begin, info.task_mask);
    size_t T_idx_end = calc_idxbits(T_end, info.task_mask);

    size_t M1_idx = calc_idxbits(M_begin, info.a_mask);
    size_t M2_idx = calc_idxbits(M_begin, info.b_mask);
    size_t N1_idx = sub_idx(info.a_mask, M1_idx, info.a_mask);
    size_t N2_idx = sub_idx(info.b_mask, M2_idx, info.b_mask);
    size_t M1_idx_end = calc_idxbits(M_end, info.a_mask);
    size_t a_msb = 1ULL << (63 - __builtin_clzll(info.a_mask));

    do
    {
        // cout << "M1: " << bitset<QUBIT>(M1_idx);
        // cout << ", N1: " << bitset<QUBIT>(N1_idx) << endl;
        do_swap_row_subtask(buffer, info, T_idx, M1_idx, M2_idx);
        do_swap_row_subtask(buffer, info, T_idx, N1_idx, N2_idx);

        M1_idx = inc_idx(M1_idx, info.a_mask);
        M2_idx = inc_idx(M2_idx, info.b_mask);
        N1_idx = dec_idx(N1_idx, info.a_mask);
        N2_idx = dec_idx(N2_idx, info.b_mask);
        if (M1_idx & a_msb)
        {
            T_idx = inc_idx(T_idx, info.task_mask);
            M1_idx = M2_idx = 0;
            N1_idx = info.a_mask;
            N2_idx = info.b_mask;
        }
    } while (T_idx != T_idx_end || M1_idx != M1_idx_end);
}

void execSQS(Complex *buffer, sqs_info_t& info) {
    size_t num_tasks = 1ULL << (info.task_bits + info.sw_bits - 1);

	#pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int tidx = omp_get_thread_num();

        /* Get Task Range */
        size_t chunk_size = num_tasks / num_threads;
        size_t remaining = num_tasks - (chunk_size * num_threads);

        /* Our Task */
        size_t task = chunk_size * tidx;
        do_swap_task_range(buffer, info, task, task + chunk_size);

        if (remaining && (size_t)tidx <= remaining)
        {
            task = (chunk_size * num_threads) + tidx;
            do_swap_task_range(buffer, info, task, task + 1);
        }
    }
}


void initSQSInfo(Qureg qureg, sqs_info_t& info, std::vector<int>& targs) {
    int qubits = qureg.numQubitPerRank;
    size_t sqsSize = targs.size();
    size_t a_mask = 0, b_mask = 0;
    // printf("SQS %lu ", sqsSize);
    for (int i = qureg.numQubitPerChunk-sqsSize; i < qureg.numQubitPerChunk; i++) {
        a_mask |= 1ULL << i;
        // printf("%u ", i);
    }
    for (size_t i = 0; i < sqsSize; i++) {
        b_mask |= 1ULL << targs[i];
        // printf("%u ", targs[i]);
    }
    // printf("\n");
    size_t sub_a_mask = 0, sub_b_mask = 0, sub_task_mask = 0;
    size_t task_mask = ((1ULL << qubits) - 1) ^ a_mask ^ b_mask;
    assert(!(a_mask & b_mask));
    assert(__builtin_popcountll(a_mask) == __builtin_popcountll(b_mask));

    /* bits */
    int sw_bits = __builtin_popcountll(a_mask);
    int task_bits = qubits - 2 * sw_bits;
    int sub_sw_bits = 0, sub_task_bits = 0;

    /* For cache friendly */
    if (sw_bits > 2)
        sub_sw_bits = std::max(__builtin_popcountll(a_mask & 3),
                        __builtin_popcountll(b_mask & 3));
    sub_task_bits = __builtin_popcountll(task_bits & ~(a_mask | b_mask) & 3);

    sub_a_mask = calc_idxbits((1ULL << sub_sw_bits) - 1, a_mask);
    sub_b_mask = calc_idxbits((1ULL << sub_sw_bits) - 1, b_mask);
    sub_task_mask = calc_idxbits((1ULL << sub_task_bits) - 1, task_mask);
    a_mask ^= sub_a_mask;
    b_mask ^= sub_b_mask;
    task_mask ^= sub_task_mask;

    sw_bits -= sub_sw_bits;
    task_bits -= sub_task_bits;

    info.a_mask = a_mask;
    info.b_mask = b_mask;
    info.task_mask = task_mask;
    info.sub_a_mask = sub_a_mask;
    info.sub_b_mask = sub_b_mask;
    info.sub_task_mask = sub_task_mask;
    info.sw_bits = sw_bits;
    info.sub_sw_bits = sub_sw_bits;
    info.task_bits = task_bits;
    info.sub_task_bits = sub_task_bits;
}