import pydantic.v1 as pydantic

from algutils.io.format import *
from algutils.param.models import YamlModel


def test_yaml_hash_exclude():

    hex1, hex2 = 'hex1', 'hex2'
    hide = 'hide'

    class A(YamlModel, hash_exclude=hex1):
        hex1: int = 55
        hide: str = 'Me1'

    class M(YamlModel, hash_exclude=hex2):
        sub: A = A()
        x: int = 0
        hex2: int = 20
        hide: str = 'Me2'

    class N(M):
        pass

    for x, C in enumerate([M, N]):
        m = C(x=x)
        dd = m.dict()
        assert hex2 in dd
        assert hex1 in dd['sub']

        # exclude hash_excludes hex* from dict and yaml_inside
        with m.exclude_context('hash') as exclude:
            dd = m.dict(exclude=exclude)
            yml_inside = m.to_yaml(exclude=exclude)
        yml_outside = m.to_yaml()

        # all hash_excludes must be out
        assert hex2 not in dd
        assert hex1 not in dd['sub']
        assert 'hex' not in yml_inside

        for yml in [yml_outside, yml_inside]:
            assert hide in yml

        # Exclude specifically hide_me from all the sub-models
        with m.exclude_context(hide) as exclude:
            dd = m.dict(exclude=exclude)
            yml_inside = m.to_yaml(exclude=exclude)
        yml_outside = m.to_yaml()

        # hash_excludes must remain
        assert hex2 in dd
        assert hex1 in dd['sub']

        #
        assert hide not in yml_inside

        assert hide in yml_outside
        assert 'Me1' in yml_outside
        assert 'Me2' in yml_outside


def test_yaml_model_finalize(tmp_path):
    # Custom YamlModel
    class MyModel(YamlModel):
        class NestA(YamlModel):
            class NestNestA(BaseModel):
                name: str = 'Bob'
                value: int = 0

            class NestNestB(BaseModel):
                age: int = 120

            sa: NestNestA = NestNestA()
            sb: NestNestB = NestNestB()
        sub: NestA = pydantic.Field(default_factory=NestA, example='Here')
        id: pydantic.conint(gt=0)

    MyModel.write_templates(tmp_path)
    assert Path(tmp_path, 'my_model_yaml_scheme.json').is_file()
    assert Path(tmp_path, 'my_model_default.yml').is_file()

    # Existing DataSource ResourceModel (skip when toolbox not installed)
    pytest.importorskip("toolbox.datacast.models")
    from toolbox.datacast.models import DataSourceRM
    scheme, default = DataSourceRM.write_templates(tmp_path)
    assert scheme.is_file()
    assert default.is_file()


if __name__ == '__main__':
    from tempfile import gettempdir
    tmp_dir = gettempdir()
    print(f'Using {tmp_dir=}')
    test_yaml_model_finalize(Path(tmp_dir))

