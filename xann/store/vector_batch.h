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
#include <turbo/container/span.h>
#include <turbo/utility/status.h>

namespace xann {
    class VectorBatch {
    public:
        VectorBatch() = default;

        ~VectorBatch();

        VectorBatch(const VectorBatch &other) = delete;

        VectorBatch &operator=(const VectorBatch &other) = delete;

        VectorBatch(VectorBatch &&other)  = default;
        VectorBatch &operator=(VectorBatch &&other) = default;

        [[nodiscard]] size_t capacity() const {
            return _capacity;
        }

        [[nodiscard]] turbo::Status init(std::size_t vector_byte_size, std::size_t n);

        [[nodiscard]] turbo::span<uint8_t> at(size_t index) const;

        void clear(size_t index);

        void set(size_t index, turbo::span<uint8_t> value);

        turbo::span<uint8_t> data() const {
            return turbo::span<uint8_t>(_data, _capacity * _vector_byte_size);
        }

        turbo::span<uint8_t> data() {
            return turbo::span<uint8_t>(_data, _capacity * _vector_byte_size);
        }
    private:
        uint64_t _vector_byte_size{0};
        uint64_t _capacity{0};
        uint8_t *_data{nullptr};
    };
} // namespace xann
