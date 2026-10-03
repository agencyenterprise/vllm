# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path

import pytest


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip the oracle unless it has the pytest session to itself.

    It turns on batch-invariant mode, which cannot be turned off and would
    change the numerics of every suite after it, and suites before it leave
    in-process engines resident: with the e2e suite first, Gemma 4 TP=1
    copies stopped being exact and async Llama TP=1 could not get its GPU
    memory.

    Like any conftest hook, this sees every item in the session, including
    other directories'; that is the point here, since any other suite's
    engines count. ``trylast`` so that ``-k``/``-m``/``--deselect`` have
    already removed what will not run. A directory-wide run therefore skips
    the oracle; run ``pytest tests/v1/kv_surgery/test_oracle.py`` as its own
    step.
    """
    oracle = Path(__file__).with_name("test_oracle.py")
    if all(item.path == oracle for item in items):
        return
    skip = pytest.mark.skip(reason="run test_oracle.py in a pytest session of its own")
    for item in items:
        if item.path == oracle:
            item.add_marker(skip)
