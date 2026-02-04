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

#include <turbo/utility/status.h>
#include <xann/store/id_manager.h>
#include <turbo/log/logging.h>

namespace xann {

    void IdManager::initialize(std::vector<LabelEntity> &&map, uint64_t reserved_id,uint64_t next_id) {
        if (_initialized) {
            return;
        }
        _ids = std::move(map);
        _next_id = next_id;
        _reserved_id = reserved_id;
        if (_ids.size() < _next_id) {
            resize(_next_id + kDefaultGrowth);
        }
        for (auto i = _reserved_id; i < _next_id; i++) {
            if (_ids[i].label == kInvalidId) {
                _free_ids.insert(i);
            } else {
                _id_map[_ids[i].label] = i;
            }
        }
        _initialized = true;
    }

    void IdManager::resize(size_t n) {
        static const LabelEntity le;
        if (n > _ids.size()) {
            _ids.resize(n, le);
        }
    }

    void IdManager::grow(size_t n) {
        static const LabelEntity le;
        auto old = _ids.size();
        _ids.resize(old + n, le);
    }

    turbo::Result<uint64_t> IdManager::alloc_id(uint64_t label) {
        KCHECK(_initialized)<<"must call initialize() first";
        if (_id_map.find(label) != _id_map.end()) {
            return turbo::already_exists_error("id already exists: ", label);
        }
        uint64_t lid;
        if (!_free_ids.empty()) {
            auto it = _free_ids.begin();
            lid = *it;
            _free_ids.erase(it);
        } else {
            if (_next_id >= _ids.size()) {
                return turbo::resource_exhausted_error("no enough id to allocate: ", _next_id);
            }
            lid = _next_id++;
        }
        _id_map[label] = lid;
        _ids[lid].label = label;
        return lid;
    }

    void IdManager::free_id(uint64_t label) {
        KCHECK(_initialized)<<"must call initialize() first";
        auto it = _id_map.find(label);
        if (it == _id_map.end()) {
            return;
        }
        auto lid = it->second;
        _id_map.erase(it);
        if (lid >= _ids.size()) {
            return;
        }
        _ids[lid].label = kInvalidId;
        _ids[lid].status = LabelEntity::kNoneStatus;
        _free_ids.insert(lid);
        shrink_next_id();
    }

    void IdManager::free_local_id(uint64_t lid) {
        KCHECK(_initialized)<<"must call initialize() first";
        if (lid >= _ids.size()) {
            return;
        }
        _id_map.erase(_ids[lid].label);
        _ids[lid].label = kInvalidId;
        _ids[lid].status = LabelEntity::kNoneStatus;
        _free_ids.insert(lid);
        shrink_next_id();
    }

    void IdManager::shrink_next_id() {
        while (_next_id > _reserved_id) {
            auto it = _free_ids.find(_next_id - 1);
            if (it == _free_ids.end()) {
                break;
            }
            _free_ids.erase(it);
            _next_id--;
        }
    }

    turbo::Result<uint64_t> IdManager::label_status(uint64_t label) const {
        auto it = _id_map.find(label);
        if (it == _id_map.end()) {
            return turbo::resource_exhausted_error("id not found: ", label);
        }
        return local_id_status(it->second);
    }

    turbo::Result<uint64_t> IdManager::local_id_status(uint64_t lid) const {
        KCHECK(_initialized)<<"must call initialize() first";
        if (lid >= _ids.size()) {
            return turbo::resource_exhausted_error("id not found: ", lid);
        }
        return _ids[lid].status;
    }

    void IdManager::set_label_status(uint64_t label, uint64_t status) {
        auto it = _id_map.find(label);
        if (it == _id_map.end()) {
            return;
        }
        set_local_id_status(it->second, status);
    }

    void IdManager::set_local_id_status(uint64_t lid, uint64_t status) {
        KCHECK(_initialized)<<"must call initialize() first";
        if (lid >= _ids.size()) {
            return;
        }
        _ids[lid].status = status;
    }

}  // namespace xann
