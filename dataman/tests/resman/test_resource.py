from pathlib import Path
from typing import Literal

import pytest

from iad.dataman.resman import ModelsManager, locatable, _log, ResourceModel, ResNameError
from iad.core import logs

logs.setup_logs(name_from=__file__,
                debug=('datacast', 'resman'),
                info=('datacast.cache', 'format'))

test_data = Path(__file__).parent / 'data'


@locatable(folders=test_data, skip_under_match=False)
class DeploymentModel(ResourceModel, desc='', patterns=['.dply.yml']):
    name: str
    description: str
    platform: Literal['linux', 'windows']


def test_scan_resources():
    _log.setLevel('DEBUG')
    dep_cm = ModelsManager.get(DeploymentModel)

    with pytest.raises(ResNameError):
        dep_cm.discover(fail=True)

    dep_cm.discover()
    assert len(dep_cm) == 2  # 2 resources with same name!


if __name__ == '__main__':
    test_scan_resources()
