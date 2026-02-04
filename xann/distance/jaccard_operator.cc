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

#include <xann/distance/jaccard_operator.h>

namespace xann {

    float simple_jaccard_distance(const turbo::span<uint8_t> &a, const turbo::span<uint8_t> &b) {
        const uint64_t *pa = reinterpret_cast<const uint64_t *>(a.data());
        const uint64_t *pb = reinterpret_cast<const uint64_t *>(b.data());
        const uint64_t *last = pa + a.size() / sizeof(uint64_t);
        const uint64_t *lastgroup = last - 3;
        size_t count = 0;
        size_t countDe = 0;
        while (pa < lastgroup) {
            count += turbo::popcount(pa[0] & pb[0]) + turbo::popcount(pa[1] & pb[1]) + turbo::popcount(pa[2] & pb[2]) + turbo::popcount(pa[3] & pb[3]);
            countDe += turbo::popcount(pa[0] | pb[0]) + turbo::popcount(pa[1] | pb[1]) + turbo::popcount(pa[2] | pb[2]) + turbo::popcount(pa[3] | pb[3]);

            pa += 4;
            pb += 4;
        }
        while (pa < last) {
            const uint64_t a_val = *pa;
            const uint64_t b_val = *pb;
            count += turbo::popcount(a_val & b_val);
            countDe += turbo::popcount(a_val | b_val);
            pa++;
            pb++;
        }

        if (countDe == 0) {
            return 0.0f;
        }

        return 1.0f - static_cast<float>(count) / static_cast<float>(countDe);
    }

    static turbo::Status initialize_l0_jaccard_operator(MetricRegistry &r) {
        ////////////////////////////////////////
        /// SimdLevel::SIMD_NONE
        /// uint8
        {
            OperatorEntity u8;
            u8.supports = true;
            u8.need_normalize_vector = false;
            u8.simd_level = SimdLevel::SIMD_NONE;
            u8.metric = kJaccard;
            u8.data_type = DataType::DT_UINT8;
            u8.normalize_vector = nullptr;
            u8.distance_vector = simple_jaccard_distance;
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


    static turbo::Status initialize_sse2_jaccard_operator(MetricRegistry &r) {
#ifdef XSIMD_WITH_SSE3
        ////////////////////////////////////////
        /// f32
        {
            OperatorEntity f32;
            f32.supports = true;
            f32.need_normalize_vector = false;
            f32.simd_level = SimdLevel::SIMD_SSE2;
            f32.metric = kJaccard;
            f32.data_type = DataType::DT_FLOAT;
            f32.normalize_vector = nullptr;
            f32.distance_vector = simd_distance_jaccard<xsimd::sse3>;
            f32.norm_vector = nullptr;

            auto rs = register_metric_level_operator(r, f32, false);
            if (!rs.ok()) {
                return rs;
            }
        }
#endif
        return turbo::OkStatus();
    }

    static turbo::Status initialize_avx2_jaccard_operator(MetricRegistry &r) {
#ifdef XSIMD_WITH_AVX2
        ////////////////////////////////////////
        /// f32
        {
            OperatorEntity f32;
            f32.supports = true;
            f32.need_normalize_vector = false;
            f32.simd_level = SimdLevel::SIMD_AVX2;
            f32.metric = kJaccard;
            f32.data_type = DataType::DT_FLOAT;
            f32.normalize_vector = nullptr;
            f32.distance_vector = simd_distance_jaccard<xsimd::avx2>;
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
        auto rs = initialize_l0_jaccard_operator(r);
        if (!rs.ok()) {
            return rs;
        }

        rs = initialize_sse2_jaccard_operator(r);
        if (!rs.ok()) {
            return rs;
        }

        rs = initialize_avx2_jaccard_operator(r);
        if (!rs.ok()) {
            return rs;
        }

        return turbo::OkStatus();
    }
}  // namespace xann
