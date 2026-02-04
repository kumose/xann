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

    float simple_jaccard_distance(const turbo::span<uint8_t> &a, const turbo::span<uint8_t> &b);

    template<typename ARCH>
    float simd_distance_jaccard(const turbo::span<uint8_t> &a, const turbo::span<uint8_t> &b) {
        using b_type = xsimd::batch<uint64_t, ARCH>;
        std::size_t inc = b_type::size;
        std::size_t size = a.size() / sizeof(uint64_t);
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;
        const uint64_t *pa = reinterpret_cast<const uint64_t *>(a.data());
        const uint64_t *pb = reinterpret_cast<const uint64_t *>(b.data());
        float sum = 0.0;
        float sum_de = 0.0;
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type avec = b_type::load(pa + i, xsimd::aligned_mode());
            b_type bvec = b_type::load(pb + i, xsimd::aligned_mode());
            sum += PopCount<ARCH>::count(avec & bvec);
            sum_de += PopCount<ARCH>::count(avec | bvec);
        }

        for (std::size_t i = vec_size; i < size; ++i) {
            auto a = pa[i] ;
            auto b = pb[i] ;
            sum += turbo::popcount(a & b);
            sum_de += turbo::popcount(a | b);
        }
        if (sum_de == 0.0) {
            return 0.0;
        }
        return 1 - sum/sum_de;
    }

    turbo::Status initialize_jaccard_operator(MetricRegistry &r);
}  // namespace xann
