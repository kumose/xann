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
    float simple_normalized_cosine_distance(const turbo::span<uint8_t> &a, const turbo::span<uint8_t> &b) {
        return simple_ip_distance<T>(a, b);
    }

    template<typename ARCH>
    float simd_normalized_cosine_distance(const turbo::span<uint8_t> &a, const turbo::span<uint8_t> &b) {
        return simd_distance_ip<ARCH>(a, b);
    }
    turbo::Status initialize_normalized_cosine_operator(MetricRegistry &r);
} // namespace xann
