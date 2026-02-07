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

#include <xann/distance/hamming_operator.h>

namespace xann {
    float simple_distance_hamming(const turbo::span<uint8_t> &a, const turbo::span<uint8_t> &b) {
        auto *last = reinterpret_cast<const uint32_t *>(a.data() + a.size());
        auto *uinta = reinterpret_cast<const uint32_t *>(a.data());
        auto *uintb = reinterpret_cast<const uint32_t *>(b.data());
        size_t count = 0;
        while (uinta < last) {
            count += turbo::popcount(*uinta++ ^ *uintb++);
        }

        return static_cast<float>(count);
    }

    static turbo::Status initialize_l0_hamming_operator(MetricRegistry &r) {
        ////////////////////////////////////////
        /// SimdLevel::SIMD_NONE
        /// uint8
        {
            OperatorEntity u8;
            u8.supports = true;
            u8.need_normalize_vector = false;
            u8.simd_level = SimdLevel::SIMD_NONE;
            u8.metric = kHamming;
            u8.data_type = DataType::DT_UINT8;
            u8.normalize_vector = nullptr;
            u8.distance_vector = simple_distance_hamming;
            u8.norm_vector = nullptr;

            auto rs = register_metric_level_operator(r, u8, false);
            if (!rs.ok()) {
                return rs;
            }
        }
        ////////////////////////////////////////
        /// SimdLevel::SIMD_NONE
        return turbo::OkStatus();
    }


    static turbo::Status initialize_sse2_hamming_operator(MetricRegistry &r) {
#ifdef XSIMD_WITH_SSE3
        ////////////////////////////////////////
        /// f32
        {
            OperatorEntity f32;
            f32.supports = true;
            f32.need_normalize_vector = false;
            f32.simd_level = SimdLevel::SIMD_SSE2;
            f32.metric = kHamming;
            f32.data_type = DataType::DT_FLOAT;
            f32.normalize_vector = nullptr;
            f32.distance_vector = simd_distance_hamming<xsimd::sse3>;
            f32.norm_vector = nullptr;

            auto rs = register_metric_level_operator(r, f32, false);
            if (!rs.ok()) {
                return rs;
            }
        }
#endif
        return turbo::OkStatus();
    }

    static turbo::Status initialize_avx2_hamming_operator(MetricRegistry &r) {
#ifdef XSIMD_WITH_AVX2
        ////////////////////////////////////////
        /// f32
        {
            OperatorEntity f32;
            f32.supports = true;
            f32.need_normalize_vector = false;
            f32.simd_level = SimdLevel::SIMD_AVX2;
            f32.metric = kHamming;
            f32.data_type = DataType::DT_FLOAT;
            f32.normalize_vector = nullptr;
            f32.distance_vector = simd_distance_hamming<xsimd::avx2>;
            f32.norm_vector = nullptr;

            auto rs = register_metric_level_operator(r, f32, false);
            if (!rs.ok()) {
                return rs;
            }
        }
#endif
        return turbo::OkStatus();
    }


    turbo::Status initialize_hamming_operator(MetricRegistry &r) {
        auto rs = initialize_l0_hamming_operator(r);
        if (!rs.ok()) {
            return rs;
        }

        rs = initialize_sse2_hamming_operator(r);
        if (!rs.ok()) {
            return rs;
        }

        rs = initialize_avx2_hamming_operator(r);
        if (!rs.ok()) {
            return rs;
        }

        return turbo::OkStatus();
    }
} // namespace xann
