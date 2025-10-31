from takeout.io import TakeoutFile
from contextlib import closing


def test_suffix():
    assert ".bar" == TakeoutFile("/a/b/c/foo.bar", "").suffix()


def test_temporary_file():
    afile = "tests/test_imagefile.py"
    img = TakeoutFile(afile, "")
    with img.open() as fp:
        assert b"from " == fp.read(5)
    with img.asfile() as tmp:
        assert tmp.filename().endswith(".py")
        assert tmp.filename() != afile
        with open(tmp.filename(), "rb") as fp:
            assert b"from " == fp.read(5)
