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