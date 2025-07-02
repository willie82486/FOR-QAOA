#ifndef _SQSGATE_H_
#define _SQSGATE_H_

#include <vector>
#include <cstdint>
#include <cstddef>
#include <cassert>
#include "state.h"

typedef struct
{
    int qubits;
    int sw_bits;
    int task_bits;
    int sub_sw_bits;
    int sub_task_bits;

    size_t a_mask;
    size_t b_mask;
    size_t sub_a_mask;
    size_t sub_b_mask;
    size_t task_mask;
    size_t sub_task_mask;
} sqs_info_t;

size_t calc_idxbits(size_t n, size_t mask);
size_t inc_idx(size_t n, size_t mask);
size_t dec_idx(size_t n, size_t mask);
size_t sub_idx(size_t a, size_t b, size_t mask);
void do_swap_block(Complex* buffer, sqs_info_t &info, size_t idx1_base, size_t idx2_base, bool A_eq_B);
void do_swap_row_subtask(Complex* buffer, sqs_info_t &info, size_t T_idx, size_t A1_idx, size_t A2_idx);
void do_swap_task_range(Complex* buffer, sqs_info_t &info, size_t begin, size_t end);
void execSQS(Complex *buffer, sqs_info_t& info);
void initSQSInfo(Qureg qureg, sqs_info_t& info, std::vector<int>& targs);

#endif