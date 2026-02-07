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

#include <vector>
#include <shared_mutex>
#include <xann/store/vector_batch.h>
#include <xann/store/id_manager.h>
#include <xann/core/vector_space.h>
#include <xann/core/option.h>

namespace xann {
    using StoreStatus = uint64_t;

    static constexpr StoreStatus kTombstone = 1;

    class  Serializer;

    class MemStore {
    public:
        MemStore(const MemStore &) = delete;

        MemStore &operator=(const MemStore &) = delete;

        turbo::Status init(const VectorSpace *vs, const VectorStoreOption &option);

        [[nodiscard]] const VectorSpace *get_vector_space() const;

        [[nodiscard]] const std::vector<VectorBatch> &vector_batch() const;

        /// add vector
        turbo::Result<uint64_t> add_vector(uint64_t snapshot_id, uint64_t label, turbo::span<uint8_t> vector);

        /// modify vector
        turbo::Result<uint64_t> set_vector(uint64_t snapshot_id, uint64_t label, turbo::span<uint8_t> vector);

        void remove_vector_by_label(uint64_t snapshot_id, uint64_t label);

        void remove_vector_by_id(uint64_t snapshot_id, uint64_t id);

        void tombstone_vector_by_label(uint64_t snapshot_id, uint64_t label);

        void tombstone_vector_by_id(uint64_t snapshot_id, uint64_t id);

        turbo::Result<uint64_t> get_label(uint64_t id) const;

        turbo::Result<uint64_t> get_id(uint64_t label) const;

        turbo::Result<turbo::span<uint8_t> > get_vector_by_label(uint64_t label) const;

        turbo::Result<turbo::span<uint8_t> > get_vector_by_id(uint64_t id) const;

        [[nodiscard]] uint64_t size() const;

        [[nodiscard]] uint64_t bytes_size() const;

        /// memory manager ment
        ///
        uint64_t allocated_bytes() const;

        uint64_t free_bytes() const;

        uint64_t allocated_vector_size() const;

        uint64_t free_vector_size() const;

        uint64_t tombstones() const;

        std::vector<uint64_t> tombstone_local_ids() const;

        std::vector<uint64_t> tombstone_labels() const;

        std::shared_mutex &mutex() const {
            return _mutex;
        }

        [[nodiscard]] uint64_t snapshot_id() const {
            return _snapshot_id;
        }

    private:
        turbo::Result<turbo::span<uint8_t> > ensure_space(uint64_t lid);

        MemStore() = default;

        ~MemStore() = default;

        friend class Serializer;
    private:
        const VectorSpace *_vector_space{nullptr};
        std::vector<VectorBatch> _vector_batches;
        std::unique_ptr<IdManager> _id_manager;
        VectorStoreOption _option;
        mutable std::shared_mutex _mutex;
        uint64_t _snapshot_id{0};
    };
} // namespace xann
