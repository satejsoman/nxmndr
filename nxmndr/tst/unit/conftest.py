# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Automatically mark tests in this directory as unit tests.

Any test file collected under tst/unit will get the pytest 'unit' marker
so they can be selectively run via '-m unit' or excluded with '-m "not unit"'.
"""

import pytest


def pytest_collection_modifyitems(session, config, items):
    for item in items:
        if "/tst/unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
