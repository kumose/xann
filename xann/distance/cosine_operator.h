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
#include <xann/distance/popcount.h>


namespace xann {

    template<typename T>
    float simple_cosine_distance(const turbo::span<uint8_t> &a, const turbo::span<uint8_t> &b) {
        const T *pa = reinterpret_cast<const T *>(a.data());
        const T *pb = reinterpret_cast<const T *>(b.data());
        const T *last = pa + a.size() / sizeof(T);
        const T *lastgroup = last - 3;
        float pa0, pa1, pa2, pa3;
        float pb0, pb1, pb2, pb3;
        float sum = 0.0;
        float norm_a = 0.0;
        float norm_b = 0.0;
        while (pa < lastgroup) {
            pa0 = pa[0];
            pa1 = pa[1];
            pa2 = pa[2];
            pa3 = pa[3];
            pb0 = pb[0];
            pb1 = pb[1];
            pb2 = pb[2];
            pb3 = pb[3];
            norm_a += pa0 * pa0 + pa1 * pa1 + pa2 * pa2 + pa3 * pa3;
            norm_b += pb0 * pb0 + pb1 * pb1 + pb2 * pb2 + pb3 * pb3;
            sum += pa0 * pb0 + pa1 * pb1 + pa2 * pb2 + pb3 * pb3;
            pa += 4;
            pb += 4;
        }
        while (pa < last) {
            pa0 = *pa++;
            pb0 = *pb++;
            norm_a += pa0 * pa0;
            norm_b += pb0 * pb0;
            sum += pa0 * pb0;
        }

        if (norm_a == 0.0 || norm_b == 0.0) {
            return 0.0;
        }
        float cosine = sum / sqrt(norm_a * norm_b);
        return cosine;
    }

    template<typename ARCH>
    float simd_distance_cosine(const turbo::span<uint8_t> &a, const turbo::span<uint8_t> &b) {
        using b_type = xsimd::batch<float, ARCH>;
        std::size_t inc = b_type::size;
        std::size_t size = a.size() / sizeof(float);
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;
        b_type sum_v = b_type::broadcast(0.0);
        b_type norm_a = b_type::broadcast(0.0);
        b_type norm_b = b_type::broadcast(0.0);
        const float *pa = reinterpret_cast<const float *>(a.data());
        const float *pb = reinterpret_cast<const float *>(b.data());
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type avec = b_type::load(pa + i, xsimd::aligned_mode());
            b_type bvec = b_type::load(pb + i, xsimd::aligned_mode());
            norm_a += avec * avec;
            norm_b += bvec * bvec;
            sum_v += avec * bvec;
        }
        auto sum = xsimd::reduce_add(sum_v);
        auto norma = xsimd::reduce_add(norm_a);
        auto normb = xsimd::reduce_add(norm_b);
        for (std::size_t i = vec_size; i < size; ++i) {

            auto ai = pa[i];
            auto bi = pb[i];
            sum += ai * bi;
            norma += ai * ai;
            normb += bi * bi;
        }
        if (norma == 0.0 || normb == 0.0) {
            return 0.0;
        }
        float cosine = sum / sqrt(norma * normb);
        return cosine;
    }

    turbo::Status initialize_cosine_operator(MetricRegistry &r);
}  // namespace xann
