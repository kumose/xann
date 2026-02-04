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

namespace xann {

    using MetricType = int32_t;

    /// none defined metric
    static constexpr MetricType kUndefinedMetric = 0;

    static constexpr MetricType kL1 = 1;

    static constexpr MetricType kL2 = 2;

    static constexpr MetricType kIP = 3;

    static constexpr MetricType kHamming = 4;

    static constexpr MetricType kJaccard = 5;

    static constexpr MetricType kCosine = 6;

    static constexpr MetricType kAngle = 7;

    static constexpr MetricType kNormalizedL2 = 8;

    static constexpr MetricType kNormalizedCosine = 9;

    static constexpr MetricType kNormalizedAngle = 10;

    static constexpr MetricType kPoincare = 11;

    static constexpr MetricType kLorentz = 12;
    static constexpr MetricType kMetricTypeMax = 30;

}  // namespace xann
