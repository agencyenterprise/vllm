# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV surgery: rare, out-of-band edits of live paged KV caches.

Physical slot order is logical order, positions are per slot, and nothing
else is special. Attention kernels are never modified; an edit is a block
table change plus ``gather`` / ``rotate`` passes over the cache between
scheduler steps.
"""

from vllm.v1.kv_surgery.kernels import apply_edit_plan, gather_slots, rotate_slots
from vllm.v1.kv_surgery.rope_descriptor import (
    RopeDescriptor,
    derive_rope_descriptors,
    inv_freq_of,
)
from vllm.v1.kv_surgery.view import EditPlan, View, drop, restrict, shift

__all__ = [
    "EditPlan",
    "RopeDescriptor",
    "View",
    "apply_edit_plan",
    "derive_rope_descriptors",
    "drop",
    "gather_slots",
    "inv_freq_of",
    "restrict",
    "rotate_slots",
    "shift",
]
