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

#include <xann/store/vector_batch.h>
#include <xann/core/vector_space.h>
#include <xsimd/memory/xsimd_aligned_allocator.hpp>

namespace xann {

    VectorBatch::~VectorBatch() {
        if (_data) {
            xsimd::aligned_allocator<uint8_t, VectorSpace::kAlignmentBytes> allocator;
            allocator.deallocate(_data, _capacity * _vector_byte_size);
            _data = nullptr;
        }
    }

    [[nodiscard]] turbo::Status VectorBatch::init(std::size_t vector_byte_size, std::size_t n) {

        try {
            xsimd::aligned_allocator<uint8_t, VectorSpace::kAlignmentBytes> allocator;
            _data = allocator.allocate(vector_byte_size * n );
            _capacity = n;
        } catch (std::exception& e) {
            return turbo::unavailable_error(e.what());
        }
        return turbo::OkStatus();
    }

    [[nodiscard]] turbo::span<uint8_t> VectorBatch::at(size_t index) const {
        if (index >= _capacity) {
            return turbo::span<uint8_t>{};
        }
        return turbo::span<uint8_t>(_data + index * _vector_byte_size, _vector_byte_size);
    }

    void VectorBatch::clear(size_t index) {
        if (index >= _capacity) {
            return;
        }
        memset(_data + index * _vector_byte_size, 0, _vector_byte_size);
    }

    void VectorBatch::set(size_t index, turbo::span<uint8_t> value) {
        if (index >= _capacity) {
            return;
        }
        memcpy(_data + index * _vector_byte_size, value.data(), _vector_byte_size);
    }
}  // namespace xann
