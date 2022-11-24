// Copyright (c) 2020, Intel CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef HOROVOD_BF16_H_
#define HOROVOD_BF16_H_

#include <stdint.h>

// #if __AVX512BF16__
// #if __AVX__ && __F16C__
#include <cpuid.h>
#include <immintrin.h>
// #endif

#define OMPI_SKIP_MPICXX
#include "mpi.h"

namespace horovod {
namespace common {

// inline void HalfBits2Float(unsigned short* src, float* res) {

// }

// float32 -> bfloat16
inline void convert_fp32_to_bf16(const void* src, void* dst) {
  __m512i y = _mm512_bsrli_epi128(_mm512_loadu_si512(src), 2);
  _mm256_storeu_si256((__m256i*)(dst), _mm512_cvtepi32_epi16(y));
}

// bfloat16 -> float32
inline void convert_bf16_to_fp32(const void* src, void* dst) {
  __m512i y = _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i const*)src));
  _mm512_storeu_si512(dst, _mm512_bslli_epi128(y, 2));
}

// execute summation over two bfloat16 numbers
void bf16_sum(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype);

} // namespace common

} // namespace horovod

#endif // HOROVOD_BF16_H_
