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
#include <cstdint>
#include <cstddef>
#include <xann/core/metric.h>
#include <turbo/container/span.h>
#include <turbo/utility/status.h>
#include <xsimd/xsimd.hpp>
#include <xann/core/operator_registry.h>

namespace xann {

    struct VectorSpace {
        static constexpr size_t kAlignmentBytes{64};
        int32_t dim{0};
        MetricType metric{kUndefinedMetric};
        DataType data_type{DataType::DT_NONE};
        int32_t alignment_dim{0};
        int32_t vector_byte_size{0};
        int32_t alignment_bytes{0};
        int32_t element_size{0};
        bool need_normalize_vector{false};
        std::string arch_name;
        xsimd::aligned_allocator<uint8_t> allocator;

        static turbo::Result<VectorSpace> create(int dim, MetricType metric, DataType dt, SimdLevel level = SimdLevel::SIMD_NONE);

        /// allocate n vector, bytes = n * alignment_dim * sizeof(DataType)
        turbo::span<uint8_t> align_allocate_vector(size_t n);

        /// allocate n vector, bytes = n * dim * sizeof(DataType)
        /// this rarely using, only for debug may have useful
        turbo::span<uint8_t> allocate_vector(size_t n);

        /// allocate n vector, bytes = n * sizeof(DataType)
        /// some case like pq, allocate some indexing
        turbo::span<uint8_t> align_allocate(size_t n);

        /// allocate n vector, bytes = n * sizeof(DataType)
        /// this rarely using, only for debug may have useful
        turbo::span<uint8_t> allocate(size_t n);

        void free(turbo::span<uint8_t> v);

        static bool is_aligned(turbo::span<uint8_t> v);

        OperatorEntity standard_operation;

        OperatorEntity operation;

    private:
        VectorSpace() = default;
    };

} // namespace xann
