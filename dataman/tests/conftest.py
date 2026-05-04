import pytest

import os
from pathlib import Path

_TEST_DIR = Path(__file__).resolve().parent

@pytest.fixture(scope='session', autouse=True)
def env_locs():
    """
    Prepare data location for testing:
        1. Set environment locators for tests
        2. Return updated EnvLoc

    Canonical dotenv: dataman/tests/test.env (also pre-set via ALG_DOTENV_FILE for algutils import-time bootstrap).
    """
    from iad.dataman.env import EnvLoc    
    os.environ["TESTS"] = str(_TEST_DIR)
    EnvLoc.reset(_TEST_DIR / "test.env", override=True)
    return EnvLoc


@pytest.fixture(scope='session')
def tiny_stereo(env_locs):
    from iad.dataman.datacast.collect import DataCollection
    from iad.dataman.factories import create_caster
    caster = create_caster('tiny')
    return DataCollection(datasets=caster, cache=False)
