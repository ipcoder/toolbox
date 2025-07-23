from pathlib import Path

import pytest
from pydantic import validator

from toolbox.resman.resource import ModelsManager, locatable, _log, ResourceModel, ResNameError
from toolbox.utils import logs

logs.setup_logs(name_from=__file__,
                debug=('datacast', 'resman'),
                info=('datacast.cache', 'format'))

test_data = Path(__file__).parent / 'data'


@locatable(folders=test_data, skip_under_match=False)
class DeploymentModel(ResourceModel, desc='', patterns=['.dply.yml']):
    name: str
    description: str
    platform: str

    @validator('platform')
    def validate_platform(cls, v):
        assert v in {'linux', 'windows'}
        return v


def test_scan_resources():
    _log.setLevel('DEBUG')
    dep_cm = ModelsManager.get(DeploymentModel)

    with pytest.raises(ResNameError):
        dep_cm.discover(fail=True)

    dep_cm.discover()
    assert len(dep_cm) == 2  # 2 resources with same name!


if __name__ == '__main__':
    test_scan_resources()
