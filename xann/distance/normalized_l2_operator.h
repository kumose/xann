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
#include <xann/distance/ip_operator.h>
#include <xann/distance/l2_operator.h>

namespace xann {
    template<typename T>
    float simple_normalized_l2_distance(const turbo::span<uint8_t> &a, const turbo::span<uint8_t> &b) {
        float v = 2.0 - 2.0 * simple_ip_distance<T>(a, b);
        if (v < 0.0) {
            return 0.0;
        }
        return sqrt(v);
    }

    template<typename T>
    void simple_normalize_l2(const turbo::span<uint8_t> &input, turbo::span<uint8_t> &output) {
        auto norm = simple_l2_norm<T>(input);
        if (norm == 0.0) {
            std::memset(output.data(), 0, output.size());
            return;
        }
        auto pin = reinterpret_cast<T *>(input.data());
        auto pout = reinterpret_cast<T *>(output.data());
        auto pend = reinterpret_cast<T *>(input.data() + input.size());
        while (pin != pend) {
            *pout++ = *pin++ / norm;
        }
    }

    template<typename ARCH>
    float simd_normalized_l2_distance(const turbo::span<uint8_t> &a, const turbo::span<uint8_t> &b) {
        float v = 2.0 - 2.0 * simd_distance_ip<ARCH>(a, b);
        if (v < 0.0) {
            return 0.0;
        }
        return sqrt(v);
    }

    template<typename ARCH>
    void simd_normalize_l2(const turbo::span<uint8_t> &input, turbo::span<uint8_t> &output) {
        auto norm = simd_norm_l2<ARCH>(input);
        if (norm == 0.0) {
            std::memset(output.data(), 0, output.size());
            return;
        }

        using b_type = xsimd::batch<float, ARCH>;
        std::size_t inc = b_type::size;
        std::size_t size = output.size()/sizeof(float);
        auto arr = reinterpret_cast<const float*>(input.data());
        auto dst = reinterpret_cast<b_type*>(output.data());
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type avec = b_type::load(&arr[i], xsimd::aligned_mode());
            avec /=  norm;
            avec.store(&dst[i], xsimd::aligned_mode());
        }
        for (std::size_t i = vec_size; i < size; ++i) {
            dst[i] = arr[i]/norm;
        }
    }

    turbo::Status initialize_normalized_l2_operator(MetricRegistry &r);
} // namespace xann
