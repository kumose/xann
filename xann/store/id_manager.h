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

#include <turbo/container/btree_set.h>
#include <turbo/container/flat_hash_map.h>
#include <vector>
#include <turbo/utility/status.h>
#include <turbo/log/logging.h>

namespace xann {
    struct LabelEntity;
    //////////////////////////////////////////////////////////////////////////
    ///
    /// @brief  Manages the mapping between external unique labels and local sequential IDs (lid) with efficient allocation/reuse.
    ///
    /// @details  Core Memory Layout (Logical Segmentation):
    ///           ```
    ///           |<-------------------------- Total ID Pool (_ids) -------------------------->|
    ///           |--- Reserved Range ---|-------- Active/Free Range --------|---- Pre-allocated Unused Range ----|
    ///           [0, _reserved_id)      [_reserved_id, _next_id)            [_next_id, _ids.size())
    ///
    ///           1. [0, _reserved_id):   Fixed reserved ID range, NOT available for any allocation.
    ///                                   This range is locked and will never be modified or used.
    ///           2. [_reserved_id, _next_id):  Active working range, contains two types of IDs:
    ///                                          - In-use IDs: Mapped to external labels (stored in _id_map, label != kInvalidId).
    ///                                          - Free IDs: Unused (marked as kInvalidId) and available for reuse (stored in _free_ids).
    ///           3. [_next_id, _ids.size()):  Pre-allocated but inactive ID range.
    ///                                          Will be activated when _next_id increments (no new memory allocation needed temporarily).
    ///           ```
    ///
    ///           Key Features:
    ///           - Prioritize reusing free IDs in _free_ids to avoid ID pool bloat.
    ///           - Automatically shrink _next_id when the trailing IDs are freed (keep pool compact).
    ///           - Support fast mapping between label (external) and lid (local) via hash map.
    ///           - Encapsulate label and custom business status via LabelEntity for logical state management.
    ///
    ///           Double-Line Control Mechanism (Core Usage for Vector Engine):
    ///           This class enables a decoupled, two-layer control system to guarantee both storage efficiency and index performance,
    ///           which is critical for vector engines with complex index structures (e.g., intermediate link vectors).
    ///           ```
    ///           Layer 1: Storage Layout Control (Physical Hole Management) <-- Managed by IdManager core
    ///           Layer 2: Index Hole Control (Logical Zombie Vector Management) <-- Enabled by LabelEntity::status, managed by outer layer
    ///           ```
    ///
    ///           1. Layer 1: Storage Layout Control (Physical Efficiency)
    ///              - **Control Object**: Physical lid slots in _ids (vector<LabelEntity>), focus on "physical holes".
    ///              - **Definition of Physical Hole**: Lid slots with label == kInvalidId (freed via free_id()/free_local_id()).
    ///              - **Core Problem**: Excessive physical holes cause memory bloat, low ID reuse efficiency, and fragmented storage.
    ///              - **Judgment Criterion**: Hole Ratio = _free_ids.size() / (_next_id - _reserved_id) (exceed preset threshold e.g., 0.3).
    ///              - **Solution**: Rebuild a new IdManager instance, only reserve valid physical IDs (label != kInvalidId) to eliminate physical holes.
    ///              - **Core Value**: Guarantee compact physical storage, reduce memory waste, and maintain high performance of ID allocation/reuse.
    ///              - **Key Note**: Does NOT care about LabelEntity::status, only focuses on the validity of label (physical state).
    ///
    ///           2. Layer 2: Index Hole Control (Logical Performance)
    ///              - **Control Object**: Logical vector data and index structures, focus on "logical zombie vectors".
    ///              - **Definition of Zombie Vector**: Lid slots with label != kInvalidId but custom status marked as "logically deleted" (e.g., hidden from business).
    ///              - **Core Problem**: Excessive zombie vectors cause index bloat, redundant retrieval overhead, and low query performance.
    ///              - **Judgment Criterion**: Zombie Ratio = Count of logically deleted vectors / Count of valid physical vectors (exceed preset threshold e.g., 0.2).
    ///              - **Solution**: Rebuild the vector index, only reserve vectors with normal status (defined by outer layer) to eliminate index holes.
    ///              - **Core Value**: Guarantee efficient index retrieval, reduce redundant overhead, and maintain the integrity of complex index structures.
    ///              - **Key Note**: Does NOT affect physical storage layout, only filters logical valid data via LabelEntity::status (business state).
    ///
    ///           3. Synergy of Two Layers
    ///              - **Independent Trigger**: Two layers can be triggered asynchronously and independently without mutual interference.
    ///              - **Asynchronous Switch**: Both reconstruction processes support hot swap (old instance serves normally during reconstruction) to ensure high availability.
    ///              - **Mutual Empowerment**: Compact physical storage (Layer 1) accelerates index reconstruction (Layer 2), and clean index (Layer 2) reduces outer layer status filtering overhead.
    ///
    class IdManager {
    public:
        /// @brief  Invalid ID marker for free local ID slots in _ids.
        /// @details  Set to the maximum value of uint64_t to avoid conflict with valid sequential IDs.
        static constexpr uint64_t kInvalidId = std::numeric_limits<uint64_t>::max();

        /// @brief  Default growth step size for expanding the ID pool (_ids) when needed.
        /// @details  Used in initialize() and grow() to pre-allocate additional ID slots efficiently.
        static constexpr uint64_t kDefaultGrowth = 256;

        /// @brief  Default constructor, initializes an empty uninitialized IdManager instance.
        IdManager() = default;

        /// @brief  Default destructor, cleans up internal containers automatically.
        ~IdManager() = default;

        /// @brief  Disable copy constructor to avoid duplicate ID mapping conflicts.
        /// @details  IdManager holds unique one-to-one mapping between labels and lids, copying is semantically invalid.
        IdManager(IdManager &) = delete;

        /// @brief  Disable copy assignment operator to avoid duplicate ID mapping conflicts.
        /// @details  IdManager holds unique one-to-one mapping between labels and lids, copy assignment is semantically invalid.
        IdManager &operator=(IdManager &) = delete;

        /// @brief  Initialize the IdManager with existing ID map, reserved ID and next ID.
        /// @param  map  Rvalue reference of existing lid-to-label map, ownership is moved to internal _ids (avoids deep copy).
        /// @param  reserved_id  Upper bound of the reserved ID range ([0, reserved_id) is locked).
        /// @param  next_id  Upper bound of the active ID range ([reserved_id, next_id) is active/workable).
        /// @note   This method can only be called once (marked as initialized after execution).
        /// @note   Automatically expands _ids to fit next_id + kDefaultGrowth if current map size is insufficient.
        turbo::Status initialize(std::vector<LabelEntity> &&map, uint64_t reserved_id, uint64_t next_id);

        /// @brief  Resize the internal ID pool (_ids) to the specified size (only supports expansion).
        /// @param  n  Target size of the ID pool.
        /// @note   Newly added slots are initialized with kInvalidId (marked as free).
        /// @note   Does nothing if n is less than or equal to the current size of _ids.
        void resize(size_t n);

        /// @brief  Grow the internal ID pool (_ids) by appending n new slots.
        /// @param  n  Number of new slots to add to the ID pool.
        /// @note   Newly added slots are initialized with kInvalidId (marked as free).
        void grow(size_t n);

        /// @brief  Allocate a local ID (lid) for the given external label.
        /// @param  label  Unique external label to map to a local ID.
        /// @return  turbo::Result<uint64_t>  Success: allocated local ID (lid); Failure: error status (e.g., label already exists).
        /// @note   Prioritizes reusing free IDs from _free_ids before allocating a new ID from _next_id.
        /// @note   Requires the IdManager to be initialized first.
        turbo::Result<uint64_t> alloc_id(uint64_t label);

        /// @brief  Free the local ID (lid) corresponding to the given external label.
        /// @param  label  External label whose mapped local ID needs to be freed.
        /// @note   Does nothing if the label does not exist in _id_map.
        /// @note   Freed ID is added to _free_ids and _next_id is shrunk if possible (via shrink_next_id()).
        void free_id(uint64_t label);

        /// @brief  Free the specified local ID (lid) directly.
        /// @param  lid  Local ID (lid) to be freed (parameter name is legacy, actual is lid).
        /// @note   Does nothing if the given lid is out of the range of _ids.
        /// @note   Freed ID is added to _free_ids and _next_id is shrunk if possible (via shrink_next_id()).
        void free_local_id(uint64_t lid);

        /// @brief  Get the current next ID (upper bound of the active ID range).
        /// @return  Current value of _next_id (next new ID to be allocated if no free IDs are available).
        [[nodiscard]] uint64_t next_id() const {
            return _next_id;
        }

        /// @brief  Get the reserved ID (upper bound of the reserved ID range).
        /// @return  Current value of _reserved_id (lock boundary of the reserved ID range).
        [[nodiscard]] uint64_t reserved_id() const {
            return _reserved_id;
        }

        /// @brief  Get the const reference of the internal lid-to-label map (_ids).
        /// @return  Const reference to _ids (avoids deep copy and prevents external modification).
        /// @note   Each slot is either a valid label or kInvalidId (free slot).
        [[nodiscard]] const std::vector<LabelEntity> &ids() const {
            return _ids;
        }

        /// @brief  Get the const reference of the internal label-to-lid hash map (_id_map).
        /// @return  Const reference to _id_map (avoids deep copy and prevents external modification).
        /// @note   Enables fast lookup of local ID (lid) by external label.
        [[nodiscard]] const turbo::flat_hash_map<uint64_t, uint64_t> &id_map() const {
            return _id_map;
        }

        /// @brief  Get the const reference of the internal free ID set (_free_ids).
        /// @return  Const reference to _free_ids (avoids deep copy and prevents external modification).
        /// @note   Sorted set of free local IDs (lids) available for reuse (ordered for efficient allocation).
        [[nodiscard]] const turbo::btree_set<uint64_t> &free_ids() const {
            return _free_ids;
        }

        /// @brief  Set an external label mapping for a local ID (lid) within the reserved range.
        /// @details  This method is exclusively used to modify the reserved ID range ([0, _reserved_id)).
        ///           The reserved range is locked for normal allocation, so this method provides the only way to set its label mapping.
        /// @param  lid  Local ID to set the label for, must be less than _reserved_id (in the reserved range).
        /// @param  label  External label to map to the specified reserved local ID (lid).
        /// @note   Triggers KCHECK failure if lid >= _reserved_id (out of reserved range).
        /// @note   Overwrites the existing mapping for the given lid (if any) in _id_map.
        void set_reserved_id(uint64_t lid, uint64_t label) {
            KCHECK(lid < _reserved_id);
            _id_map[lid] = label;
        }

        turbo::Result<uint64_t> local_id(uint64_t label) const;

        /// @brief  Query the business status of the given external label.
        /// @param  label  External unique label to query status for.
        /// @return  turbo::Result<uint64_t>  Success: the business status of the label; Failure: error status (e.g., label not found).
        turbo::Result<LabelEntity> label_entity(uint64_t label) const;

        /// @brief  Query the business status of the given local ID (lid).
        /// @param  lid  Local ID to query status for.
        /// @return  turbo::Result<uint64_t>  Success: the business status of the local ID; Failure: error status (e.g., lid out of range).
        turbo::Result<LabelEntity> local_entity(uint64_t lid) const;

        /// @brief  Modify the business status of the given external label.
        /// @param  label  External unique label to modify status for.
        /// @param  status  New custom business status to set (defined by the outer layer).
        /// @note   Does nothing if the label does not exist in _id_map.
        void set_label_status(uint64_t label, uint64_t status);

        /// @brief  Modify the business status of the given local ID (lid).
        /// @param  lid  Local ID to modify status for.
        /// @param  status  New custom business status to set (defined by the outer layer).
        /// @note   Does nothing if the lid is out of the range of _ids.
        void set_local_id_status(uint64_t lid, uint64_t status);
    private:
        /// @brief  Shrink _next_id to the smallest possible value by checking trailing free IDs.
        /// @details  Iteratively decrements _next_id if the trailing ID (next_id - 1) is in _free_ids.
        /// @note   Stops when _next_id reaches _reserved_id (will not shrink into the reserved range).
        /// @note   Cleans up the trailing free ID from _free_ids after shrinking _next_id.
        void shrink_next_id();

    private:
        /// @brief  Sorted set of free local IDs (lids) available for reuse.
        /// @details  Uses btree_set for efficient ordered lookup, insertion and deletion (O(logN) complexity).
        turbo::btree_set<uint64_t> _free_ids;

        /// @brief  Core lid-to-label mapping vector (the entire ID pool).
        /// @details  Follows the logical segmentation layout, slots are initialized with kInvalidId if free.
        /// @note   Supports O(1) random access to the label of a given local ID (lid).
        std::vector<LabelEntity> _ids;

        /// @brief  Next new local ID to be allocated (upper bound of the active ID range).
        /// @details  Increments when no free IDs are available for reuse.
        uint64_t _next_id{0};

        /// @brief  Upper bound of the reserved ID range ([0, _reserved_id) is locked and unavailable).
        /// @details  Acts as the lower bound of the active ID range ([_reserved_id, _next_id)).
        uint64_t _reserved_id{0};

        /// @brief  Hash map for fast label-to-lid mapping.
        /// @details  Uses flat_hash_map for higher space efficiency and faster access than std::unordered_map.
        /// @note   Enables O(1) lookup of local ID (lid) by external label.
        turbo::flat_hash_map<uint64_t, uint64_t> _id_map;

        /// @brief  Initialization status flag.
        /// @details  Prevents uninitialized usage and duplicate initialization of the IdManager.
        /// @note   Set to true only after successful execution of initialize().
        bool _initialized{false};
    };

    /// @brief  Entity struct encapsulating external label and its associated business status.
        /// @details  Used as the core element of the ID pool (_ids), each instance binds a local ID (lid) with its label and status.
    struct LabelEntity {
        /// @brief  Default none status for free/inactive LabelEntity.
        /// @details  Set to 0, used to reset status when a local ID is freed.
        static constexpr uint64_t kNoneStatus = 0;

        /// @brief  External unique label (marked as IdManager::kInvalidId if the local ID is free).
        uint64_t label{IdManager::kInvalidId};

        /// @brief  Custom business status of the label (defined and interpreted by the outer layer).
        /// @details  Reset to kNoneStatus when the local ID is freed, supports 64-bit custom flags (bitwise operation is allowed).
        uint64_t status{kNoneStatus};
    };
} // namespace xann
