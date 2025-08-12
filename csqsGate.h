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
 
#ifndef _CSQSGATE_H_
#define _CSQSGATE_H_

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#ifdef USE_MPI
  #include <mpi.h>
#endif
#include "state.h"

#ifdef USE_MPI
  #define MPI_CHECK(x)                                                           \
    do {                                                                         \
      int __ret = (x);                                                           \
      if (MPI_SUCCESS != __ret) {                                                \
        char err_string[MPI_MAX_ERROR_STRING];                                   \
        int err_string_len = 0;                                                  \
        MPI_Error_string(__ret, err_string, &err_string_len);                    \
        fprintf(stderr, "(%s:%d) ERROR: MPI call returned error code %d (%s)",   \
                __FILE__, __LINE__, __ret, err_string);                          \
      }                                                                          \
    } while (0)
#endif

uint64_t __attribute__((always_inline)) inline insert_bit_0(uint64_t task, const char targ);
uint64_t __attribute__((always_inline)) inline insert_bits_0(uint64_t task, const char targs[]);
#ifdef USE_MPI
  void CSQS(Qureg& qureg, const int csqsSize, char* targ);
  void CSQS_MultiThread(Qureg& qureg, const int csqsSize, char* targ);
#endif

#endif