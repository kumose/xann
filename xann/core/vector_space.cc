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

#include <xann/core/vector_space.h>
#include <xann/core/operator_registry.h>
#include <xann/distance/l1_operator.h>
#include <xann/distance/l2_operator.h>
#include <turbo/container/flat_hash_map.h>

namespace xann {
    turbo::Result<int32_t> data_type_size(DataType dt) {
        switch (dt) {
            case DataType::DT_UINT8:
                return sizeof(uint8_t);
            case DataType::DT_FLOAT16:
                return sizeof(uint16_t);
            case DataType::DT_FLOAT:
                return sizeof(float);
            default:
                return turbo::invalid_argument_error("unknown datatype");
        }
    }

    turbo::Result<VectorSpace> VectorSpace::create(int dim, MetricType metric, DataType dt, SimdLevel level) {
        VectorSpace vs;
        vs.dim = dim;
        vs.metric = metric;
        vs.data_type = dt;
        vs.alignment_bytes = xsimd::aligned_allocator<uint8_t, kAlignmentBytes>::alignment;

        auto dtrs = data_type_size(vs.data_type);
        if (!dtrs.ok()) {
            return dtrs.status();
        }
        vs.element_size = dtrs.value_or_die();

        vs.vector_byte_size = (vs.element_size * vs.dim + vs.alignment_bytes - 1) / vs.alignment_bytes * vs.
                              alignment_bytes;
        vs.alignment_dim = vs.vector_byte_size / vs.element_size;

        /// standary
        auto msrs = MetricRegistry::instance().get_metric_operator(vs.metric, vs.data_type, SimdLevel::SIMD_NONE);
        if (!msrs.ok()) {
            return msrs.status();
        }
        vs.standard_operation  = msrs.value_or_die();

        auto mrs = MetricRegistry::instance().get_metric_operator(vs.metric, vs.data_type, level);
        if (!mrs.ok()) {
            return mrs.status();
        }
        auto ms = mrs.value_or_die();
        if (!ms.supports) {
            return turbo::unavailable_error("not supported");
        }
        vs.operation  = ms;
        vs.need_normalize_vector = ms.need_normalize_vector;
        vs.arch_name = xsimd::default_arch::name();
        /// check params valid
        return vs;
    }

    turbo::span<uint8_t> VectorSpace::align_allocate_vector(size_t n) {
        auto nalloc = static_cast<size_t>(n * vector_byte_size);
        auto ptr = allocator.allocate(nalloc);
        return turbo::span<uint8_t>{ptr, nalloc};
    }

    turbo::span<uint8_t> VectorSpace::allocate_vector(size_t n) {
        auto nalloc = static_cast<size_t>(n * element_size * dim);
        auto ptr = allocator.allocate(nalloc);
        return turbo::span<uint8_t>{ptr, nalloc};
    }

    turbo::span<uint8_t> VectorSpace::align_allocate(size_t n) {
        auto nalloc = static_cast<size_t>(n * element_size);
        auto ptr = allocator.allocate(nalloc);
        return turbo::span<uint8_t>{ptr, nalloc};
    }

    turbo::span<uint8_t> VectorSpace::allocate(size_t n) {
        auto nalloc = static_cast<size_t>(n * element_size);
        auto ptr = allocator.allocate(nalloc);
        return turbo::span<uint8_t>{ptr, nalloc};
    }

    void VectorSpace::free(turbo::span<uint8_t> v) {
        allocator.deallocate(v.data(), v.size());
    }

    bool VectorSpace::is_aligned(turbo::span<uint8_t> v) {
        return xsimd::is_aligned(v.data());
    }

} // namespace xann
