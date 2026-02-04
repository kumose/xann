// Copyright (C) Kumo inc. and its affiliates.
// Author: Jeff.li lijippy@163.com
// All rights reserved.
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//

#pragma once

#include <turbo/container/span.h>
#include <cmath>
#include <xann/common/half.hpp>
#include <xann/core/operator_registry.h>
#include <xann/core/vector_space.h>
#include <xsimd/xsimd.hpp>

namespace xann {

    template<typename T>
    float simple_ip_distance(const turbo::span<uint8_t> &a, const turbo::span<uint8_t> &b) {
        const T *pa = reinterpret_cast<const T *>(a.data());
        const T *pb = reinterpret_cast<const T *>(b.data());
        const T *last = pa + a.size() / sizeof(T);
        const T *lastgroup = last - 3;
        float diff0, diff1, diff2, diff3;
        float d = 0.0;
        while (pa < lastgroup) {
            diff0 = static_cast<float>(pa[0] * pb[0]);
            diff1 = static_cast<float>(pa[1] * pb[1]);
            diff2 = static_cast<float>(pa[2] * pb[2]);
            diff3 = static_cast<float>(pa[3] * pb[3]);
            d += diff0  + diff1  + diff2  + diff3 ;
            pa += 4;
            pb += 4;
        }
        while (pa < last) {
            diff0 = static_cast<float>(*pa++ * *pb++);
            d += diff0 ;
        }
        return sqrt(d);
    }

    template<typename ARCH>
    float simd_distance_ip(const turbo::span<uint8_t> &a, const turbo::span<uint8_t> &b) {
        using b_type = xsimd::batch<float, ARCH>;
        std::size_t inc = b_type::size;
        std::size_t size = a.size() / sizeof(float);
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;
        b_type sum_v = b_type::broadcast(0.0);
        const float *pa = reinterpret_cast<const float *>(a.data());
        const float *pb = reinterpret_cast<const float *>(b.data());
        double sum = 0.0;
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type avec = b_type::load(pa + i, xsimd::aligned_mode());
            b_type bvec = b_type::load(pb + i, xsimd::aligned_mode());
            sum_v += xsimd::mul(avec, bvec);
        }
        sum += xsimd::reduce_add(sum_v);
        for (std::size_t i = vec_size; i < size; ++i) {
            sum += pa[i] * pb[i];
        }
        return sqrt(sum);
    }

    turbo::Status initialize_ip_operator(MetricRegistry &r);
}  // namespace xann
