import pytest

from toolbox.utils.filesproc import Locator, Path


def test_locator(tmp_path):
    import os

    VAR_NAME = 'TEST_VAR'

    env = tmp_path / 'env'
    os.environ[VAR_NAME] = str(env)

    folders = [tmp_path / 'f1', tmp_path / 'f2']
    loc = Locator(folders[0], str(folders[1]), envar='TEST_VAR')
    assert loc.first_existing(tmp_path) == tmp_path

    given = {env, *folders}
    for p in given:
        p.mkdir(exist_ok=True)

    found = set(loc.existing())
    assert found == given
    assert loc.first_existing() == env
    assert loc.first == env

    # ------------ operators -------------------
    assert (loc / 'ff').first_of('I') == folders[0] / 'ff'
    assert [*(loc + (
        f := [tmp_path / 'add_f'])
              ).defined(order='I')] == folders + f

    os.environ[VAR_NAME] = 'INVALID'  # spoil to check its deals with not existing
    assert len([*loc.existing()]) == len(found) - 1
    assert loc.first_existing() == folders[0]
    assert loc.first == Path('INVALID')
    os.environ[VAR_NAME] = str(env)  # restore

    # ------------ first file --------
    name = 'part/name.ext'
    path = folders[1] / name
    path.parent.mkdir(exist_ok=True)
    path.touch(exist_ok=True)
    assert loc.first_file(name) == path

    # ------------ order
    more_folders = [tmp_path / 'f3', tmp_path / 'f4']
    assert len([*loc.existing(*more_folders, order='I')]) == 2
    assert len([*loc.existing(*more_folders, order='A')]) == 0  # not existing
    with pytest.raises(NotADirectoryError):
        _ = [*loc.existing(*more_folders, order='A', fail=True)]

    for p in more_folders:
        p.mkdir(exist_ok=True)

    assert len([*loc.existing(*more_folders, order='A')]) == 2  # now existing

    # ordering
    found = [*loc.existing(*more_folders, order='AE')]
    assert found == [*more_folders, env]

    found = [*loc.existing(*more_folders, order='EIA')]
    assert found == [env, *folders, *more_folders]

    # locator of locators
    loc2 = Locator(*more_folders, loc, envar='NOVAR')

    found = [*loc2.existing(order='EIA')]
    assert found == [*more_folders, env, *folders]

    # check unique filter
    assert found == [*loc2.existing(env, order='EIA')]
    found = [*loc2.existing(env, order='EIA', unique=False)]
    assert found == [*more_folders, env, *folders, env]

    assert not set(loc.defined()).issuperset(more_folders)
    loc += more_folders
    assert set(loc.defined()).issuperset(more_folders)


def test_locator_not_responsive(tmp_path):
    loc = Locator(tmp_path, timeout=1)  # locator with one existing folder
    assert any(loc.defined()), "Expected 1 defined"
    assert any(loc.existing()), "Expected 1 existing"

    loc.validate(alarm=True)
    assert any(loc.defined()), "Must validate in default time"

    assert not loc.safe, "Default safe must be False"
    assert any(loc.existing(safe=True)), "Must safe-find in default time"

    loc.set_check_opt(caching=False, timeout=0)
    assert not any(loc.existing(safe=True)), "Must not safe-find in 0 time"

    loc.validate(alarm=True)   # removes the "invalid" folder
    assert not any(loc.defined()), "Must have failed to validate in 0 time"

