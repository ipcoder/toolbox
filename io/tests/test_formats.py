import pytest

from toolbox.io.format import FileFormat, Content, MetaForms, MetaSrc


@pytest.mark.refactor
def test_new_format():
    initial_formats = FileFormat.find_handler(out='all')
    assert initial_formats == []

    class TestAbsFormat(FileFormat):
        pass

    assert not FileFormat.find_handler(out='all')

    with pytest.raises(NotImplementedError):
        class MissingContentFormat(FileFormat, patterns='*.txt'):
            pass

    with pytest.raises(NotImplementedError):
        class MissingReadFormat(FileFormat, patterns='*.txt', content=Content.DATA):
            pass

    class NewFormat(FileFormat, content=Content.DATA, patterns='.txt'):
        @classmethod
        def read(cls, **kws):
            pass

    cur_formats = FileFormat.find_handler(out='all')
    added = list(set(cur_formats).difference(initial_formats))
    assert len(added) == 1 and added[0] is NewFormat

    FileFormat.forget_formats()
    assert not FileFormat.find_handler()


def test_formats_types():
    class AbstractBinaryFormatWithMetaData(FileFormat, content=Content.BIN | Content.META):
        pass

    class TextFormat(FileFormat, patterns='.txt', content=Content.DATA):
        @classmethod
        def read(cls, filename, **kws):
            if kws: raise NotImplementedError
            with open(filename, 'rt') as f:
                return f.read()

    assert TextFormat.is_abstract is False
    assert AbstractBinaryFormatWithMetaData.is_abstract is True
    assert TextFormat.supports_meta() is False
    assert AbstractBinaryFormatWithMetaData.supports_meta() is True

    assert TextFormat.content is Content.DATA
    assert TextFormat.patterns == ['.txt']

    fmt = FileFormat.find_handler('ok.txt')
    assert fmt is TextFormat
    assert fmt.supports_write() is False


@pytest.mark.refactor
def test_meta():
    # from toolbox.io.basic_formats import ConfigFormat
    meta = MetaSrc('file.txt')
    # assert MetaSrc(meta)
    form = MetaForms(meta)
    assert form.sources[0] == meta
    # assert isinstance(FileFormat.find_handler('file.tml'), ConfigFormat)
    # assert MetaForms('file.tml').sources[0].reader == FileFormat.find_handler('file.tml').read
