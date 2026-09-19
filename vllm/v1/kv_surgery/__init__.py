# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV surgery: rare, out-of-band edits of live paged KV caches.

Physical slot order is logical order, positions are per slot, and nothing
else is special. Attention kernels are never modified; an edit is a block
table change plus ``gather`` / ``rotate`` passes over the cache between
scheduler steps.

Ops (``ops.py``): ``DropOp``, ``ShiftOp`` and ``SpliceOp`` go through
``edit_kv``; ``fork_kv`` starts a request on a shared prefix of another one.
Restricted prefill is a fork plus a splice; a probe is a fork with
``max_tokens=1`` and full-vocab logprobs.
"""

from vllm.v1.kv_surgery.kernels import apply_edit_plan, gather_slots, rotate_slots
from vllm.v1.kv_surgery.rope_descriptor import (
    RopeDescriptor,
    derive_rope_descriptors,
    inv_freq_of,
)
from vllm.v1.kv_surgery.view import (
    EditPlan,
    View,
    drop,
    fork,
    shift,
    splice,
)

__all__ = [
    "EditPlan",
    "RopeDescriptor",
    "View",
    "apply_edit_plan",
    "derive_rope_descriptors",
    "drop",
    "fork",
    "gather_slots",
    "inv_freq_of",
    "rotate_slots",
    "shift",
    "splice",
]
