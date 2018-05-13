/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <iomanip>

#include "ngraph/log.hpp"

float round_binary(float x, uint32_t binary_digits)
{
    union fp_t {
        float f;
        uint32_t i;
    };
    // uint32_t exponent = 0x3F000000;
    uint32_t lsb = 1 << (23 - binary_digits);
    uint32_t mask = lsb - 1;
    fp_t half;
    half.f = x;
    int8_t exp = static_cast<int8_t>((half.i & 0x7F800000) >> 23) - 127;
    half.i &= 0xFF800000;
    NGRAPH_INFO << static_cast<int32_t>(exp);
    exp -= (binary_digits + 1);
    fp_t new_fp;
    new_fp.i = static_cast<uint8_t>(exp + 127);
    new_fp.i <<= 23;
    NGRAPH_INFO << static_cast<int32_t>(exp);
    NGRAPH_INFO << std::hex << mask;
    NGRAPH_INFO << std::hex << half.i;
    NGRAPH_INFO << std::hex << new_fp.i;
    NGRAPH_INFO << new_fp.f;

    half.i |= lsb;
    NGRAPH_INFO << std::hex << half.i;
    NGRAPH_INFO << std::hex << ~mask;
    // void* tmp = static_cast<void*>(&half);
    // float* half_fp = static_cast<float*>(tmp);
    NGRAPH_INFO << std::setw(15) << std::setprecision(16) << half.f;
    // NGRAPH_INFO << std::hex << static_cast<uint32_t>(x);
    fp_t y;
    y.f = x + new_fp.f;
    NGRAPH_INFO << y.f;
    y.i &= ~mask;
    NGRAPH_INFO << y.f;
    return y.f;
}
