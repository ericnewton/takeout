import zipfile
from abc import abstractmethod, ABC
import tempfile
import mimetypes
import magic
import hashlib
import logging
import os.path
from io import UnsupportedOperation
from PIL.Image import IO
from typing import Tuple, Generator

logger = logging.getLogger(__name__)

def listing(archive: str) -> Generator[str, None, None]:
    try:
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                yield name
    except Exception:
        logger.exception("Unable to read archive %s", archive)

class Reader(ABC, IO[bytes]):
    "things that can be read and closed"

    @abstractmethod
    def read(self, size: int = -1) -> bytes:
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError()

    def seek(self, offset, how=0):
        raise UnsupportedOperation()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        return False


class FReader(Reader):
    "A file-based Reader"

    def __init__(self, fp):
        self.fp = fp

    def read(self, size: int = -1) -> bytes:
        return self.fp.read(size)

    def close(self) -> None:
        return self.fp.close()


class ZReader(Reader):
    "A Reader for an entry in a ZipFile"

    def __init__(self, entry: "InputFile"):
        self.zf = zipfile.ZipFile(entry.archive)
        self.df = self.zf.open(entry.path)

    def read(self, size: int = -1) -> bytes:
        return self.df.read(size)

    def close(self):
        self.df.close()
        self.zf.close()


class TemporaryFile(str):
    "Provide an open file reference from a InputFile in a `with` clause"

    BUFSIZE = 1024 * 1024

    def __init__(self, img: "InputFile"):
        self.op = tempfile.NamedTemporaryFile(suffix=img.suffix())
        self.img = img

    def filename(self):
        return self.op.name

    def __enter__(self) -> "TemporaryFile":
        with self.img.open() as ip:
            while True:
                buf = ip.read(self.BUFSIZE)
                if not buf:
                    break
                self.op.write(buf)
        self.op.flush()
        return self

    def __exit__(self, *exc):
        self.op.close()
        return False


class InputFile(object):
    "Track a filename, possibly inside an archive"
    UNKNOWN_TYPE = "application/octet"

    def __init__(self, path: str, archive: str):
        self.path = path
        self.archive = archive

    def open(self) -> Reader:
        if self.archive:
            return ZReader(self)
        return FReader(open(self.path, "rb"))

    def asfile(self) -> TemporaryFile:
        "if a file is in an archive, reify it as a temp file"
        return TemporaryFile(self)

    def suffix(self):
        return os.path.splitext(self.path)[1]

    def __str__(self):
        if self.archive:
            return f"{self.path} in {self.archive}"
        return self.path

    def mimetype(self) -> str:
        "guess a file type by its extension or initial bytes"
        
        fast_mimetype = mimetypes.guess_file_type(self.path)
        if fast_mimetype is None:
            return self.UNKNOWN_TYPE
        type_subtype, encoding = fast_mimetype
        if type_subtype is None:
            return self.UNKNOWN_TYPE
        if type_subtype in {"application/json"}:
            return type_subtype
        main_type = type_subtype.split("/", 1)[0]
        if main_type in {"video", "image"}:
            return type_subtype
        # known type, but maybe we should dig deeper (eg, '.3gp' files)
        with self.open() as fp:
            bytes = fp.read(1024)
            magic_type = magic.from_buffer(bytes, mime=True)
            if magic_type is None:
                return self.UNKNOWN_TYPE
            return magic_type
        
    def hash_and_size(self) -> Tuple[str, int]:
        size = 0
        with self.open() as fp:
            hasher = hashlib.sha256()
            while True:
                buf = fp.read(1024 * 1024)
                if not buf:
                    break
                size += len(buf)
                hasher.update(buf)
            digest = hasher.hexdigest()
        return digest, size


