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

#include "bf16.h"

namespace horovod {
namespace common {

// #if __AVX__ && __F16C__
#if __AVX512BF16__ && __AVX512F__ && __AVX512BW__
// Query CPUID to determine AVX512F and AVX512_BF16 runtime support.
bool is_avx512f_bf16() {
  static bool initialized = false;
  static bool result = false;
  if (!initialized) {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
      result = (eax & bit_AVX512BF16) && (ebx & bit_AVX512F) &&
               (ebx & bit_AVX512BW);
    }
    initialized = true;
  }
  return result;
}
#endif

void bf16_sum(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype) {
  // cast bfloat16 numbers to uint16_t
  auto* in_vec = (unsigned short*)invec;
  auto* inout_vec = (unsigned short*)inoutvec;

  int i = 0;
#if __AVX512BF16__ && __AVX512F__ && __AVX512BW__
  // #if __AVX__ && __F16C__
  if (is_avx512f_bf16()) {
    // set 0
    // __m512 src = 0;
    // set all bfloat numbers to 1? How to?
    // __m512bh b = 1;
    for (; i < (*len / 16) * 16; i += 16) {
      // convert in & inout to m512
      __m512i in_m512, out_m512;
      convert_bf16_to_fp32((const void*)(in_vec+i), (void*)(&in_m512));
      convert_bf16_to_fp32((const void*)(inout_vec+i), (void*)(&out_m512));
      // add them together to new_inout_m256
      __m512 newout_m512 = _mm512_add_ps((__m512)in_m512, (__m512)out_m512);
      // convert back and store in inout
      convert_fp32_to_bf16((__m512i*)(&newout_m512), (__m256i*)(inout_vec+i));
    }
  }
#endif
  // process the remaining data
  for(i = (*len / 16) * 16; i < *len; i++){
    unsigned int tmp_in = (*(in_vec + i)) << 16;
    unsigned int tmp_out = (*(inout_vec + i)) << 16;
    float in_float = *reinterpret_cast<float*>(&tmp_in);
    float inout_float = *reinterpret_cast<float*>(&tmp_out);
    inout_float += in_float;
    *(inout_vec + i) = *reinterpret_cast<unsigned int*>(&inout_float)>>16;
  }
}

} // namespace common

} // namespace horovod
