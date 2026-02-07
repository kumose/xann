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

#include <xann/store/store.h>

namespace xann {
    turbo::Status MemStore::init(const VectorSpace *vs, const VectorStoreOption &option) {
        _vector_space = vs;
        _option = option;
        _id_manager = std::make_unique<IdManager>();
        std::vector<LabelEntity> v;
        return _id_manager->initialize(std::move(v), _option.reserved, _option.reserved + 1);
    }

    const VectorSpace *MemStore::get_vector_space() const {
        return _vector_space;
    }

    const std::vector<VectorBatch> &MemStore::vector_batch() const {
        return _vector_batches;
    }

    turbo::Result<uint64_t> MemStore::add_vector(uint64_t snapshot_id,uint64_t label, turbo::span<uint8_t> vector) {
        auto rs = _id_manager->alloc_id(label);
        if (!rs.ok()) {
            return rs.status();
        }
        auto lid = rs.value_or_die();
        auto ers = ensure_space(lid);
        if (!ers.ok()) {
            return ers.status();
        }
        auto sp = ers.value_or_die();
        /// sp must not null, guard by ensure_space
        memcpy(sp.data(), vector.data(), vector.size());
        _snapshot_id = snapshot_id;
        return lid;
    }

    turbo::Result<uint64_t> MemStore::set_vector(uint64_t snapshot_id,uint64_t label, turbo::span<uint8_t> vector) {
        auto rs = _id_manager->local_id(label);
        if (!rs.ok()) {
            return rs.status();
        }
        auto lid = rs.value_or_die();
        auto bi = lid / _option.batch_size;
        auto si = lid % _option.batch_size;
        auto sp = _vector_batches[bi].at(si);
        if (sp.empty()) {
            return turbo::out_of_range_error("vector out of range, lid:", lid, " label:", label, " batch index:", si);
        }
        memcpy(sp.data(), vector.data(), vector.size());
        _snapshot_id = snapshot_id;
        return lid;
    }

    void MemStore::remove_vector_by_label(uint64_t snapshot_id, uint64_t label) {
        _id_manager->free_id(label);
        _snapshot_id = snapshot_id;
    }

    void MemStore::remove_vector_by_id(uint64_t snapshot_id,uint64_t id) {
        _id_manager->free_local_id(id);
        _snapshot_id = snapshot_id;
    }

    void MemStore::tombstone_vector_by_label(uint64_t snapshot_id,uint64_t label) {
        _id_manager->set_label_status(label, kTombstone);
        _snapshot_id = snapshot_id;
    }

    void MemStore::tombstone_vector_by_id(uint64_t snapshot_id,uint64_t id) {
        _id_manager->set_local_id_status(id, kTombstone);
        _snapshot_id = snapshot_id;
    }

    turbo::Result<uint64_t> MemStore::get_label(uint64_t id) const {
        auto rs = _id_manager->local_entity(id);
        if (!rs.ok()) {
            return rs.status();
        }
        return rs.value_or_die().label;
    }

    turbo::Result<uint64_t> MemStore::get_id(uint64_t label) const {
        return _id_manager->local_id(label);
    }

    turbo::Result<turbo::span<uint8_t> > MemStore::get_vector_by_label(uint64_t label) const {
        auto rs = _id_manager->local_id(label);
        if (!rs.ok()) {
            return rs.status();
        }
        auto lid = rs.value_or_die();
        auto bi = lid / _option.batch_size;
        auto si = lid % _option.batch_size;
        auto sp = _vector_batches[bi].at(si);
        if (sp.empty()) {
            return turbo::out_of_range_error("vector out of range, lid:", lid, " label:", label, " batch index:", si);
        }
        return sp;
    }

    turbo::Result<turbo::span<uint8_t> > MemStore::get_vector_by_id(uint64_t lid) const {
        auto bi = lid / _option.batch_size;
        auto si = lid % _option.batch_size;
        auto sp = _vector_batches[bi].at(si);
        if (sp.empty()) {
            return turbo::out_of_range_error("vector out of range, lid:", lid, " batch index:", si);
        }
        return sp;
    }

    [[nodiscard]] uint64_t MemStore::size() const {
        return _id_manager->id_map().size();
    }

    [[nodiscard]] uint64_t MemStore::bytes_size() const {
        return  _id_manager->id_map().size() * _vector_space->vector_byte_size;
    }
    uint64_t MemStore::allocated_bytes() const {
        auto n = _vector_batches.size();
        return n * _vector_space->vector_byte_size * _option.batch_size;
    }

    uint64_t MemStore::free_bytes() const {
        auto n = _id_manager->free_ids().size();
        return n * _vector_space->vector_byte_size ;
    }

    uint64_t MemStore::allocated_vector_size() const {
        auto n = _vector_batches.size();
        return n * _option.batch_size;
    }

    uint64_t MemStore::free_vector_size() const {
        auto n = _id_manager->free_ids().size();
        return n;
    }

    uint64_t MemStore::tombstones() const {
        uint64_t n = 0;
        auto &ids = _id_manager->ids();
        auto end = std::min(ids.size(), static_cast<size_t>(_id_manager->next_id()));
        for (auto i = _id_manager->reserved_id(); i < end; i++) {
            if (ids[i].status == kTombstone) {
                ++n;
            }
        }
        return n;
    }

    std::vector<uint64_t> MemStore::tombstone_local_ids() const {
        std::vector<uint64_t> lids;
        auto &ids = _id_manager->ids();
        auto end = std::min(ids.size(), static_cast<size_t>(_id_manager->next_id()));
        for (auto i = _id_manager->reserved_id(); i < end; i++) {
            if (ids[i].status == kTombstone) {
                lids.push_back(i);
            }
        }
        return lids;
    }

    std::vector<uint64_t> MemStore::tombstone_labels() const {
        std::vector<uint64_t> labels;
        auto &ids = _id_manager->ids();
        auto end = std::min(ids.size(), static_cast<size_t>(_id_manager->next_id()));
        for (auto i = _id_manager->reserved_id(); i < end; i++) {
            if (ids[i].status == kTombstone) {
                labels.push_back(ids[i].label);
            }
        }
        return labels;
    }

    turbo::Result<turbo::span<uint8_t> > MemStore::ensure_space(uint64_t lid) {
        if (lid >= _option.max_elements) {
            return turbo::out_of_range_error("lid:", lid);
        }
        auto bi = lid / _option.batch_size;
        auto si = lid % _option.batch_size;
        auto diff = bi + 1 <= _vector_batches.size() ? 0 : bi + 1 - _vector_batches.size();
        for (auto i = 0; i < diff; i++) {
            VectorBatch b;
            auto rs = b.init(_vector_space->vector_byte_size, _option.batch_size);
            if (!rs.ok()) {
                return rs;
            }
            _vector_batches.push_back(std::move(b));
        }
        return _vector_batches[bi].at(si);
    }
} // namespace xann
