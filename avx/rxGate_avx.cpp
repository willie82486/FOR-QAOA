#include "rxGate_avx.h"
#include <immintrin.h>

#define SIMD_BIT 3
#define SIMD_LEN (1 << SIMD_BIT)

inline ull bit_string(const ull task, const int target) {
    ull mask = (1ull << target) - 1;
    return ((task >> target) << (target + 1)) | (task & mask);
}

inline void sincosf64(double angle, double* sinRad, double* cosRad) {
    *sinRad = sin(angle);
    *cosRad = cos(angle);
}

void rotateX_avx(Qureg& qureg, const double angle, const int targetQubit, ull idx) {
    double sinAngle, cosAngle;
    sincosf64(angle / 2, &sinAngle, &cosAngle);

    const ull half_off = 1 << targetQubit, off = half_off * 2;

    if (half_off == 1) {
        const __m256d sinAngle_vec_256 = _mm256_set1_pd(sinAngle);
        const __m256d cosAngle_vec_256 = _mm256_set1_pd(cosAngle);

        for (ull i = 0; i < qureg.numAmpPerChunk; i += SIMD_LEN) {
            ull off0 = idx + i;
            ull off1 = off0 + half_off;
            _mm_prefetch((char*)&qureg.stateVec[off0 + 8], _MM_HINT_T0);
            _mm_prefetch((char*)&qureg.stateVec[off1 + 8], _MM_HINT_T0);

            __m512d up_r = _mm512_set_pd(
                qureg.stateVec[off0+7].real, qureg.stateVec[off0+6].real,
                qureg.stateVec[off0+5].real, qureg.stateVec[off0+4].real,
                qureg.stateVec[off0+3].real, qureg.stateVec[off0+2].real,
                qureg.stateVec[off0+1].real, qureg.stateVec[off0+0].real
            );
            __m512d up_i = _mm512_set_pd(
                qureg.stateVec[off0+7].imag, qureg.stateVec[off0+6].imag,
                qureg.stateVec[off0+5].imag, qureg.stateVec[off0+4].imag,
                qureg.stateVec[off0+3].imag, qureg.stateVec[off0+2].imag,
                qureg.stateVec[off0+1].imag, qureg.stateVec[off0+0].imag
            );
            __m512d lo_r = _mm512_set_pd(
                qureg.stateVec[off1+7].real, qureg.stateVec[off1+6].real,
                qureg.stateVec[off1+5].real, qureg.stateVec[off1+4].real,
                qureg.stateVec[off1+3].real, qureg.stateVec[off1+2].real,
                qureg.stateVec[off1+1].real, qureg.stateVec[off1+0].real
            );
            __m512d lo_i = _mm512_set_pd(
                qureg.stateVec[off1+7].imag, qureg.stateVec[off1+6].imag,
                qureg.stateVec[off1+5].imag, qureg.stateVec[off1+4].imag,
                qureg.stateVec[off1+3].imag, qureg.stateVec[off1+2].imag,
                qureg.stateVec[off1+1].imag, qureg.stateVec[off1+0].imag
            );

            __mmask8 k_up_indices = 0x55;
            __mmask8 k_lo_indices = 0xAA;
            __m256d up_r_masked = _mm512_castpd512_pd256(_mm512_maskz_compress_pd(k_up_indices, up_r));
            __m256d up_i_masked = _mm512_castpd512_pd256(_mm512_maskz_compress_pd(k_up_indices, up_i));
            __m256d lo_r_masked = _mm512_castpd512_pd256(_mm512_maskz_compress_pd(k_lo_indices, lo_r));
            __m256d lo_i_masked = _mm512_castpd512_pd256(_mm512_maskz_compress_pd(k_lo_indices, lo_i));
            __v4df up_r_new = _mm256_fmadd_pd(up_r_masked, cosAngle_vec_256, _mm256_mul_pd(lo_i_masked, sinAngle_vec_256));
            __v4df up_i_new = _mm256_fmsub_pd(up_i_masked, cosAngle_vec_256, _mm256_mul_pd(lo_r_masked, sinAngle_vec_256));
            __v4df lo_r_new = _mm256_fmadd_pd(up_i_masked, sinAngle_vec_256, _mm256_mul_pd(lo_r_masked, cosAngle_vec_256));
            __v4df lo_i_new = _mm256_fmsub_pd(lo_i_masked, cosAngle_vec_256, _mm256_mul_pd(up_r_masked, sinAngle_vec_256));

            __m512d up_r_new_512 = _mm512_zextpd256_pd512(up_r_new);
            __m512d up_i_new_512 = _mm512_zextpd256_pd512(up_i_new);
            __m512d lo_r_new_512 = _mm512_zextpd256_pd512(lo_r_new);
            __m512d lo_i_new_512 = _mm512_zextpd256_pd512(lo_i_new);

            __m512d r_new = _mm512_mask_expand_pd(_mm512_setzero_pd(), k_up_indices, up_r_new_512);
            r_new = _mm512_mask_expand_pd(r_new, k_lo_indices, lo_r_new_512);

            __m512d i_new = _mm512_mask_expand_pd(_mm512_setzero_pd(), k_up_indices, up_i_new_512);
            i_new = _mm512_mask_expand_pd(i_new, k_lo_indices, lo_i_new_512);
            
            double* rptr0  = (double*)&qureg.stateVec[off0];
            double* iptr0 = rptr0 + 1;
            for (int k = 0; k < 8; k++) {
                rptr0[k*2] = ((double*)&r_new)[k];
                iptr0[k*2] = ((double*)&i_new)[k];
            }
        }
    }else if (half_off == 2) {
        const __m256d sinAngle_vec_256 = _mm256_set1_pd(sinAngle);
        const __m256d cosAngle_vec_256 = _mm256_set1_pd(cosAngle);

        for (ull i = 0; i < qureg.numAmpPerChunk; i += SIMD_LEN) {
            ull off0 = idx + i;
            ull off1 = off0 + half_off;
            _mm_prefetch((char*)&qureg.stateVec[off0 + 8], _MM_HINT_T0);
            _mm_prefetch((char*)&qureg.stateVec[off1 + 8], _MM_HINT_T0);

            __m512d up_r = _mm512_set_pd(
                qureg.stateVec[off0+7].real, qureg.stateVec[off0+6].real,
                qureg.stateVec[off0+5].real, qureg.stateVec[off0+4].real,
                qureg.stateVec[off0+3].real, qureg.stateVec[off0+2].real,
                qureg.stateVec[off0+1].real, qureg.stateVec[off0+0].real
            );
            __m512d up_i = _mm512_set_pd(
                qureg.stateVec[off0+7].imag, qureg.stateVec[off0+6].imag,
                qureg.stateVec[off0+5].imag, qureg.stateVec[off0+4].imag,
                qureg.stateVec[off0+3].imag, qureg.stateVec[off0+2].imag,
                qureg.stateVec[off0+1].imag, qureg.stateVec[off0+0].imag
            );
            __m512d lo_r = _mm512_set_pd(
                qureg.stateVec[off1+7].real, qureg.stateVec[off1+6].real,
                qureg.stateVec[off1+5].real, qureg.stateVec[off1+4].real,
                qureg.stateVec[off1+3].real, qureg.stateVec[off1+2].real,
                qureg.stateVec[off1+1].real, qureg.stateVec[off1+0].real
            );
            __m512d lo_i = _mm512_set_pd(
                qureg.stateVec[off1+7].imag, qureg.stateVec[off1+6].imag,
                qureg.stateVec[off1+5].imag, qureg.stateVec[off1+4].imag,
                qureg.stateVec[off1+3].imag, qureg.stateVec[off1+2].imag,
                qureg.stateVec[off1+1].imag, qureg.stateVec[off1+0].imag
            );

            __mmask8 k_up_indices = 0x33;
            __mmask8 k_lo_indices = 0xCC;
            __m256d up_r_masked = _mm512_castpd512_pd256(_mm512_maskz_compress_pd(k_up_indices, up_r));
            __m256d up_i_masked = _mm512_castpd512_pd256(_mm512_maskz_compress_pd(k_up_indices, up_i));
            __m256d lo_r_masked = _mm512_castpd512_pd256(_mm512_maskz_compress_pd(k_lo_indices, lo_r));
            __m256d lo_i_masked = _mm512_castpd512_pd256(_mm512_maskz_compress_pd(k_lo_indices, lo_i));
            __v4df up_r_new = _mm256_fmadd_pd(up_r_masked, cosAngle_vec_256, _mm256_mul_pd(lo_i_masked, sinAngle_vec_256));
            __v4df up_i_new = _mm256_fmsub_pd(up_i_masked, cosAngle_vec_256, _mm256_mul_pd(lo_r_masked, sinAngle_vec_256));
            __v4df lo_r_new = _mm256_fmadd_pd(up_i_masked, sinAngle_vec_256, _mm256_mul_pd(lo_r_masked, cosAngle_vec_256));
            __v4df lo_i_new = _mm256_fmsub_pd(lo_i_masked, cosAngle_vec_256, _mm256_mul_pd(up_r_masked, sinAngle_vec_256));

            __m512d up_r_new_512 = _mm512_zextpd256_pd512(up_r_new);
            __m512d up_i_new_512 = _mm512_zextpd256_pd512(up_i_new);
            __m512d lo_r_new_512 = _mm512_zextpd256_pd512(lo_r_new);
            __m512d lo_i_new_512 = _mm512_zextpd256_pd512(lo_i_new);

            __m512d r_new = _mm512_mask_expand_pd(_mm512_setzero_pd(), k_up_indices, up_r_new_512);
            r_new = _mm512_mask_expand_pd(r_new, k_lo_indices, lo_r_new_512);

            __m512d i_new = _mm512_mask_expand_pd(_mm512_setzero_pd(), k_up_indices, up_i_new_512);
            i_new = _mm512_mask_expand_pd(i_new, k_lo_indices, lo_i_new_512);
            
            double* rptr0  = (double*)&qureg.stateVec[off0];
            double* iptr0 = rptr0 + 1;
            for (int k = 0; k < 8; k++) {
                rptr0[k*2] = ((double*)&r_new)[k];
                iptr0[k*2] = ((double*)&i_new)[k];
            }
        }
    }else if (half_off == 4) {
        const __m256d sinAngle_vec_256 = _mm256_set1_pd(sinAngle);
        const __m256d cosAngle_vec_256 = _mm256_set1_pd(cosAngle);

        for (ull i = 0; i < qureg.numAmpPerChunk; i += off) {
            for (ull j = 0; j < half_off; j += 4) {
                ull off0 = idx + i + j;
                ull off1 = off0 + half_off;
    
                _mm_prefetch((char*)&qureg.stateVec[off0 + 4], _MM_HINT_T0);
                _mm_prefetch((char*)&qureg.stateVec[off1 + 4], _MM_HINT_T0);
                __m256d up_r = _mm256_set_pd(
                    qureg.stateVec[off0+3].real, qureg.stateVec[off0+2].real,
                    qureg.stateVec[off0+1].real, qureg.stateVec[off0+0].real
                );
                __m256d up_i = _mm256_set_pd(
                    qureg.stateVec[off0+3].imag, qureg.stateVec[off0+2].imag,
                    qureg.stateVec[off0+1].imag, qureg.stateVec[off0+0].imag
                );
                __m256d lo_r = _mm256_set_pd(
                    qureg.stateVec[off1+3].real, qureg.stateVec[off1+2].real,
                    qureg.stateVec[off1+1].real, qureg.stateVec[off1+0].real
                );
                __m256d lo_i = _mm256_set_pd(
                    qureg.stateVec[off1+3].imag, qureg.stateVec[off1+2].imag,
                    qureg.stateVec[off1+1].imag, qureg.stateVec[off1+0].imag
                );
                __v4df up_r_new = _mm256_fmadd_pd(up_r, cosAngle_vec_256, _mm256_mul_pd(lo_i, sinAngle_vec_256));
                __v4df up_i_new = _mm256_fmsub_pd(up_i, cosAngle_vec_256, _mm256_mul_pd(lo_r, sinAngle_vec_256));
                __v4df lo_r_new = _mm256_fmadd_pd(up_i, sinAngle_vec_256, _mm256_mul_pd(lo_r, cosAngle_vec_256));
                __v4df lo_i_new = _mm256_fmsub_pd(lo_i, cosAngle_vec_256, _mm256_mul_pd(up_r, sinAngle_vec_256));

                __m256d up_lo_interleaved = _mm256_unpacklo_pd(up_r_new, up_i_new);
                __m256d up_hi_interleaved = _mm256_unpackhi_pd(up_r_new, up_i_new);
                _mm256_store_pd((double*)&qureg.stateVec[off0], up_lo_interleaved);
                _mm256_store_pd((double*)&qureg.stateVec[off0+2], up_hi_interleaved);

                __m256d lo_lo_interleaved = _mm256_unpacklo_pd(lo_r_new, lo_i_new);
                __m256d lo_hi_interleaved = _mm256_unpackhi_pd(lo_r_new, lo_i_new);
                _mm256_store_pd((double*)&qureg.stateVec[off1], lo_lo_interleaved);
                _mm256_store_pd((double*)&qureg.stateVec[off1+2], lo_hi_interleaved);
            }
        }
    }
    else {
        const __m512d sinAngle_vec = _mm512_set1_pd(sinAngle);
        const __m512d cosAngle_vec = _mm512_set1_pd(cosAngle);
        for (ull i = 0; i < qureg.numAmpPerChunk; i += off) {
            for (ull j = 0; j < half_off; j += SIMD_LEN) {
                ull off0 = idx + i + j;
                ull off1 = off0 + half_off;
    
                _mm_prefetch((char*)&qureg.stateVec[off0 + 8], _MM_HINT_T0);
                _mm_prefetch((char*)&qureg.stateVec[off1 + 8], _MM_HINT_T0);
                __m512d up_r = _mm512_set_pd(
                    qureg.stateVec[off0+7].real, qureg.stateVec[off0+6].real,
                    qureg.stateVec[off0+5].real, qureg.stateVec[off0+4].real,
                    qureg.stateVec[off0+3].real, qureg.stateVec[off0+2].real,
                    qureg.stateVec[off0+1].real, qureg.stateVec[off0+0].real
                );
                __m512d up_i = _mm512_set_pd(
                    qureg.stateVec[off0+7].imag, qureg.stateVec[off0+6].imag,
                    qureg.stateVec[off0+5].imag, qureg.stateVec[off0+4].imag,
                    qureg.stateVec[off0+3].imag, qureg.stateVec[off0+2].imag,
                    qureg.stateVec[off0+1].imag, qureg.stateVec[off0+0].imag
                );
                __m512d lo_r = _mm512_set_pd(
                    qureg.stateVec[off1+7].real, qureg.stateVec[off1+6].real,
                    qureg.stateVec[off1+5].real, qureg.stateVec[off1+4].real,
                    qureg.stateVec[off1+3].real, qureg.stateVec[off1+2].real,
                    qureg.stateVec[off1+1].real, qureg.stateVec[off1+0].real
                );
                __m512d lo_i = _mm512_set_pd(
                    qureg.stateVec[off1+7].imag, qureg.stateVec[off1+6].imag,
                    qureg.stateVec[off1+5].imag, qureg.stateVec[off1+4].imag,
                    qureg.stateVec[off1+3].imag, qureg.stateVec[off1+2].imag,
                    qureg.stateVec[off1+1].imag, qureg.stateVec[off1+0].imag
                );
                
                __v8df up_r_new = _mm512_fmadd_pd(up_r, cosAngle_vec, _mm512_mul_pd(lo_i, sinAngle_vec));
                __v8df up_i_new = _mm512_fmsub_pd(up_i, cosAngle_vec, _mm512_mul_pd(lo_r, sinAngle_vec));
                __v8df lo_r_new = _mm512_fmadd_pd(up_i, sinAngle_vec, _mm512_mul_pd(lo_r, cosAngle_vec));
                __v8df lo_i_new = _mm512_fmsub_pd(lo_i, cosAngle_vec, _mm512_mul_pd(up_r, sinAngle_vec));

                __m512d up_lo_interleaved = _mm512_unpacklo_pd(up_r_new, up_i_new);
                __m512d up_hi_interleaved = _mm512_unpackhi_pd(up_r_new, up_i_new);
                _mm512_store_pd((double*)&qureg.stateVec[off0], up_lo_interleaved);
                _mm512_store_pd((double*)&qureg.stateVec[off0+4], up_hi_interleaved);

                __m512d lo_lo_interleaved = _mm512_unpacklo_pd(lo_r_new, lo_i_new);
                __m512d lo_hi_interleaved = _mm512_unpackhi_pd(lo_r_new, lo_i_new);
                _mm512_store_pd((double*)&qureg.stateVec[off1], lo_lo_interleaved);
                _mm512_store_pd((double*)&qureg.stateVec[off1+4], lo_hi_interleaved);
            }
        }
    }
}

void multiRotateX_avx(Qureg& qureg, const double angle, 
                    std::vector<std::vector<int>>& GBs, bool csqsflag, std::vector<int>& qubitMap) {
    for (size_t GBsIdx = 0; GBsIdx < GBs.size(); GBsIdx++) {
#ifdef USE_MPI
        if (GBsIdx > 0 && (*GBs[GBsIdx-1].rbegin() == qureg.numQubitPerRank-1) && csqsflag) {
            int csqsSize = qureg.numQubitTotal - qureg.numQubitPerRank; // Device qubit
            char *targ = (char *)calloc(csqsSize, sizeof(char));
            for (int i = 0; i < csqsSize; i++) {
                targ[i] = i;
            }     
            CSQS_MultiThread(qureg, csqsSize, targ);
            csqsflag = false;
            for (int i = qureg.numQubitTotal-csqsSize; i < qureg.numQubitTotal; i++) {
                std::swap(qubitMap[i-csqsSize], qubitMap[i]);
            }
        }
#endif
        if (GBsIdx != 0) {
            sqs_info_t info;
            initSQSInfo(qureg, info, GBs[GBsIdx]);
            execSQS(qureg.stateVec, info);
            int swapSize = GBs[GBsIdx].size();
            for (int i = 0; i < swapSize; i++) {
                std::swap(qubitMap[qureg.numQubitPerChunk-1-i], qubitMap[GBs[GBsIdx][swapSize-1-i]]);
            }
        }   
     
        #pragma omp parallel for schedule(static)
        for (ull idx = 0; idx < qureg.numAmpPerRank; idx += qureg.numAmpPerChunk) {
            for (int i = qureg.numQubitPerChunk-GBs[GBsIdx].size() ; i < qureg.numQubitPerChunk; i++) {
                rotateX_avx(qureg, angle, GBs[0][i], idx);
            }
        }
    }
}

