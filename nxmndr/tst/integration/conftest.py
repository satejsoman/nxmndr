# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Automatically mark tests in this directory as integration.

Any test file collected under tst/integration will get the pytest 'integration' marker
so they can be selectively run via '-m integration' or excluded with '-m "not integration"'.
"""

import numpy as np
import pytest
from pathlib import Path


def pytest_collection_modifyitems(session, config, items):
    for item in items:
        # Only mark items that physically reside in the integration folder
        if "/tst/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def synthetic_tif():
    """Generate a small synthetic GeoTIFF for integration tests.

    Creates a 64x64, 3-band, uint8 raster in EPSG:3857 and writes it to
    ``tst/integration/image.tif``.  The file is created once per test session
    and cleaned up afterwards.
    """
    import rasterio
    from rasterio.transform import from_bounds

    tif_path = Path(__file__).parent / "image.tif"

    transform = from_bounds(0, 0, 64, 64, 64, 64)
    rng = np.random.default_rng(42)
    data = rng.integers(0, 256, size=(3, 64, 64), dtype=np.uint8)

    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=transform,
    ) as dst:
        dst.write(data)

    yield tif_path

    tif_path.unlink(missing_ok=True)
    aux = tif_path.with_suffix(".tif.aux.xml")
    aux.unlink(missing_ok=True)
