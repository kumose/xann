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
#include <xann/common/half.hpp>
#include <xann/core/operator_registry.h>
#include <xann/core/vector_space.h>

namespace xann {
    inline double absolute(double v) { return fabs(v); }

    inline double absolute(half_float::half v) { return fabs(v); }

    inline int absolute(int v) { return abs(v); }

    inline long absolute(long v) { return abs(v); }

    template<typename T>
    float simple_distance_l1(const turbo::span<uint8_t> &a, const turbo::span<uint8_t> &b) {
        const T *pa = reinterpret_cast<const T *>(a.data());
        const T *pb = reinterpret_cast<const T *>(b.data());
        float diff0, diff1, diff2, diff3;
        float d = 0.0;
        const T *last = reinterpret_cast<const T *>(a.data()) + a.size() / sizeof(T);
        const T *lastgroup = last - 3;
        while (pa < lastgroup) {
            diff0 = (float) (pa[0] - pb[0]);
            diff1 = (float) (pa[1] - pb[1]);
            diff2 = (float) (pa[2] - pb[2]);
            diff3 = (float) (pa[3] - pb[3]);
            d += absolute(diff0) + absolute(diff1) + absolute(diff2) + absolute(diff3);
            pa += 4;
            pb += 4;
        }
        while (pa < last) {
            diff0 = (float) *pa++ - (float) *pb++;
            d += absolute(diff0);
        }
        return d;
    }

    template<typename T>
    float simple_normal_l1(const turbo::span<uint8_t> &a) {
        const T *pa = reinterpret_cast<const T *>(a.data());
        float d = 0.0;
        const T *last = reinterpret_cast<const T *>(a.data()) + a.size() / sizeof(T);
        const T *lastgroup = last - 3;
        while (pa < lastgroup) {
            d += absolute(pa[0]) + absolute(pa[1]) + absolute(pa[2]) + absolute(pa[3]);
            pa += 4;
        }
        while (pa < last) {
            d += absolute((float) *pa++);
        }
        return d;
    }

    template<typename A = xsimd::default_arch>
    float simd_distance_l1(const turbo::span<uint8_t> &a, const turbo::span<uint8_t> &b) {
        using b_type = xsimd::batch<float, A>;
        bool is_aligned = VectorSpace::is_aligned(a) && VectorSpace::is_aligned(b);
        std::size_t inc = b_type::size;
        std::size_t size = a.size()/sizeof(float);
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;
        float *pa = reinterpret_cast<float *>(a.data());
        float *pb = reinterpret_cast<float *>(b.data());
        double sum = 0.0;
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type avec = b_type::load(pa + i, xsimd::aligned_mode());
            b_type bvec = b_type::load(pb + i, xsimd::aligned_mode());
            sum += xsimd::reduce_add(xsimd::abs(avec - bvec));
        }
        for (std::size_t i = vec_size; i < size; ++i) {
            sum += absolute(pa[i] - pb[i]);
        }
        return sum;
    }

    template<typename A = xsimd::default_arch>
    float simd_normal_l1(const turbo::span<uint8_t> &a) {
        using b_type = xsimd::batch<float, A>;
        bool is_aligned = VectorSpace::is_aligned(a);
        std::size_t inc = b_type::size;
        std::size_t size = a.size() / sizeof(float);
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;
        float *pa = reinterpret_cast<float *>(a.data());
        double sum = 0.0;
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type avec = b_type::load(pa + i, xsimd::aligned_mode());
            sum += xsimd::reduce_add(xsimd::abs(avec));
        }
        for (std::size_t i = vec_size; i < size; ++i) {
            sum += absolute(pa[i]);
        }
        return sum;
    }

    turbo::Status initialize_l1_operator(MetricRegistry &r);
} // namespace xann
