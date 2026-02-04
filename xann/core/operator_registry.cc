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

#include <xann/core/operator_registry.h>
#include <xann/distance/l1_operator.h>
#include <xann/distance/l2_operator.h>
#include <xann/distance/ip_operator.h>
#include <xann/distance/hamming_operator.h>
#include <xann/distance/jaccard_operator.h>
#include <xann/distance/cosine_operator.h>
#include <xann/distance/angle_operator.h>
#include <xann/distance/normalized_l2_operator.h>
#include <xann/distance/normalized_cosine_operator.h>
#include <xann/distance/normalized_angle_operator.h>
#include <mutex>
#include <turbo/log/logging.h>

namespace xann {
    MetricRegistry::MetricRegistry() {
        _metric_level_map.resize(kMetricTypeMax);
        auto rs = register_builtin_operator(*this);
        KCHECK(rs.ok())<<rs.to_string();
    }

    turbo::Status MetricRegistry::register_operator(OperatorEntity op, bool replace) {
        if (_finish_build) {
            return turbo::failed_precondition_error("already registered");
        }

        if (op.metric <= kUndefinedMetric || op.metric >= kMetricTypeMax) {
            return turbo::invalid_argument_error("invalid metric type:", op.metric);
        }

        auto &mit = _metric_level_map[static_cast<size_t>(op.metric)];
        if (!mit.init) {
            mit.init = true;
        }

        if (static_cast<int>(op.data_type) <= static_cast<int>(DataType::DT_NONE) || static_cast<int>(op.data_type) >=
            static_cast<int>(DataType::DT_MAX)) {
            return turbo::invalid_argument_error("invalid data type:", static_cast<int>(op.data_type));
        }
        auto &dit = mit.operators[static_cast<int>(op.data_type)];
        if (!dit.init) {
            dit.init = true;
        }

        if (static_cast<int>(op.simd_level) < static_cast<int>(SimdLevel::SIMD_NONE) || static_cast<int>(op.simd_level)
            >= static_cast<int>(SimdLevel::SIMD_MAX)) {
            return turbo::invalid_argument_error("invalid simd level:", static_cast<int>(op.simd_level));
        }

        auto &sit = dit.operators[static_cast<int>(op.simd_level)];
        if (sit.init && !replace) {
            return turbo::already_exists_error("already inited:", static_cast<int>(op.simd_level));
        }
        if (!sit.init) {
            sit.init = true;
        }
        sit.operators[static_cast<int>(op.simd_level)] = op;
        return turbo::OkStatus();
    }

    turbo::Result<OperatorEntity> MetricRegistry::get_metric_operator(MetricType metric, DataType dt,
                                                                      SimdLevel simd_level) {
        if (metric <= kUndefinedMetric || metric >= kMetricTypeMax) {
            return turbo::invalid_argument_error("invalid metric type:", metric);
        }

        auto &mit = _metric_level_map[static_cast<size_t>(metric)];
        if (!mit.init) {
            return turbo::unavailable_error("unavailable metric type:", metric);
        }

        if (static_cast<int>(dt) <= static_cast<int>(DataType::DT_NONE) || static_cast<int>(dt) >= static_cast<int>(
                DataType::DT_MAX)) {
            return turbo::invalid_argument_error("invalid data type:", static_cast<int>(dt));
        }
        auto &dit = mit.operators[static_cast<int>(dt)];
        if (!dit.init) {
            return turbo::unavailable_error("unavailable data type:", static_cast<int>(dt));
        }

        if (static_cast<int>(simd_level) < static_cast<int>(SimdLevel::SIMD_NONE) || static_cast<int>(simd_level) >=
            static_cast<int>(SimdLevel::SIMD_MAX)) {
            return turbo::invalid_argument_error("invalid simd level:", static_cast<int>(simd_level));
        }

        auto &sit = dit.operators[static_cast<int>(simd_level)];
        if (!sit.init) {
            return turbo::already_exists_error("unavailable simd level:", static_cast<int>(simd_level));
        }
        return sit.operators[static_cast<int>(simd_level)];
    }

    std::vector<OperatorEntity> MetricRegistry::all_metric_operators() {
        std::vector<OperatorEntity> result;
        for (auto &it: _metric_level_map) {
            if (!it.init) {
                continue;
            }
            for (auto &dit: it.operators) {
                if (!dit.init) {
                    continue;
                }
                for (auto &sit: dit.operators) {
                    if (!sit.init) {
                        continue;
                    }
                    for (auto &op: sit.operators) {
                        if (!op.supports) {
                            continue;
                        }
                        result.push_back(op);
                    }
                }
            }
        }
        return result;
    }


    std::once_flag bin_flag;

    turbo::Status register_builtin_operator_once(MetricRegistry &r) {
        auto rs = initialize_l1_operator(r);
        if (!rs.ok()) {
            return rs;
        }
        rs = initialize_l2_operator(r);
        if (!rs.ok()) {
            return rs;
        }
        rs = initialize_ip_operator(r);
        if (!rs.ok()) {
            return rs;
        }
        rs = initialize_hamming_operator(r);
        if (!rs.ok()) {
            return rs;
        }
        rs = initialize_jaccard_operator(r);
        if (!rs.ok()) {
            return rs;
        }
        rs = initialize_cosine_operator(r);
        if (!rs.ok()) {
            return rs;
        }
        rs = initialize_angle_operator(r);
        if (!rs.ok()) {
            return rs;
        }
        rs = initialize_normalized_l2_operator(r);
        if (!rs.ok()) {
            return rs;
        }
        rs = initialize_normalized_cosine_operator(r);
        if (!rs.ok()) {
            return rs;
        }
        rs = initialize_normalized_angle_operator(r);
        if (!rs.ok()) {
            return rs;
        }
        return turbo::OkStatus();
    }

    turbo::Status register_builtin_operator(MetricRegistry &r) {
        turbo::Status ret;
        std::call_once(bin_flag, [&ret, &r]() {
            ret = register_builtin_operator_once(r);
        });
        return ret;
    }
} // namespace xann
