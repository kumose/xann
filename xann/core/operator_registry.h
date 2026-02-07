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

#include <turbo/container/flat_hash_map.h>
#include <xann/common/half.hpp>
#include <turbo/container/span.h>
#include <turbo/utility/status.h>
#include <xann/core/metric.h>

namespace xann {
    enum class DataType {
        DT_NONE = 0,
        DT_UINT8,
        DT_FLOAT16,
        DT_FLOAT,
        DT_MAX,
    };

    turbo::Result<int32_t> data_type_size(DataType dt);

    template<DataType dt>
    struct data_type_traits {
    };

    template<>
    struct data_type_traits<DataType::DT_UINT8> {
        using value_type = uint8_t;
    };

    template<>
    struct data_type_traits<DataType::DT_FLOAT16> {
        using value_type = uint16_t;
    };

    template<>
    struct data_type_traits<DataType::DT_FLOAT> {
        using value_type = float;
    };


    typedef void (*normalize_vector_func)(const turbo::span<uint8_t> &input, turbo::span<uint8_t> &output);

    typedef float (*distance_vector_func)(const turbo::span<uint8_t> &v1, const turbo::span<uint8_t> &v2);

    typedef float (*norm_vector_func)(const turbo::span<uint8_t> &v1);


    enum class SimdLevel {
        SIMD_NONE = 0,
        SIMD_SSE2 = 1,
        SIMD_AVX2 = 2,
        SIMD_AVX512 = 3,
        SIMD_MAX = 4
    };

    struct OperatorEntity {
        /// false means this is invalid.
        bool supports{false};

        bool need_normalize_vector{false};

        SimdLevel simd_level{SimdLevel::SIMD_NONE};

        MetricType metric{kUndefinedMetric};

        DataType data_type{DataType::DT_NONE};

        normalize_vector_func normalize_vector{nullptr};

        distance_vector_func distance_vector{nullptr};

        norm_vector_func norm_vector{nullptr};
    };

    struct SimdLevelMap {
        SimdLevelMap() {
            operators.resize(static_cast<size_t>(SimdLevel::SIMD_MAX));
        }
        bool init{false};
        std::vector<OperatorEntity> operators;
    };

    struct DataTypeMap {
        DataTypeMap() {
            operators.resize(static_cast<size_t>(DataType::DT_MAX));
        }
        bool init{false};
        std::vector<SimdLevelMap> operators;
    };

    struct  MetricLevelMap {
        MetricLevelMap() {
            operators.resize(kMetricTypeMax);
        }
        bool init{false};
        std::vector<DataTypeMap> operators;
    };

    class MetricRegistry {
    public:
        static MetricRegistry &instance() {
            static MetricRegistry registry;
            return registry;
        }

        turbo::Result<OperatorEntity> get_metric_operator(MetricType metric, DataType dt, SimdLevel simd_level);

        turbo::Status register_operator(OperatorEntity op, bool replace = false);

        /// mark end of building,
        /// MetricRegistry is immutable now.
        void finish_build() {
            _finish_build = true;
        }

        std::vector<OperatorEntity> all_metric_operators();


    private:
        MetricRegistry();

        bool _finish_build{false};
        std::vector<MetricLevelMap> _metric_level_map;
    };

    inline turbo::Status register_metric_level_operator(MetricRegistry &r, OperatorEntity op, bool replace = false) {
        return r.register_operator(op, replace);
    }

    turbo::Status register_builtin_operator(MetricRegistry &r);
}
