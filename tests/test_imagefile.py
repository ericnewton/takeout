from takeout.io import InputFile
from contextlib import closing


def test_suffix():
    assert ".bar" == InputFile("/a/b/c/foo.bar", "").suffix()


def test_temporary_file():
    afile = "tests/test_imagefile.py"
    img = InputFile(afile, "")
    with img.open() as fp:
        assert b"from " == fp.read(5)
    with img.asfile() as tmp:
        assert tmp.filename().endswith(".py")
        assert tmp.filename() != afile
        with open(tmp.filename(), "rb") as fp:
            assert b"from " == fp.read(5)
