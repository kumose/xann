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

#include <xsimd/xsimd.hpp>

namespace xann {

    template <typename ARCH>
    struct PopCount {
        static constexpr std::size_t size = xsimd::batch<uint64_t, ARCH>::size;

       static float count(const xsimd::batch<uint64_t, ARCH> &batch) {
            if constexpr(size == 2) {
                /// 2 * sizeof(uint64_t) * 8 = 128
                return turbo::popcount(batch.get(0)) + turbo::popcount(batch.get(1));
            } else if constexpr(size == 4) {
                /// 4 * sizeof(uint64_t) * 8 = 256
                return turbo::popcount(batch.get(0)) + turbo::popcount(batch.get(1)) + turbo::popcount(batch.get(2)) + turbo::popcount(batch.get(3));
            } else if constexpr (size == 8) {
                /// 8 * sizeof(uint64_t) * 8 = 512
                return turbo::popcount(batch.get(0)) + turbo::popcount(batch.get(1)) + turbo::popcount(batch.get(2)) + turbo::popcount(batch.get(3))
                        + turbo::popcount(batch.get(4)) + turbo::popcount(batch.get(5)) + turbo::popcount(batch.get(6)) + turbo::popcount(batch.get(7));
            }
            TURBO_UNREACHABLE();
        }
    };


}  // namespace xann
