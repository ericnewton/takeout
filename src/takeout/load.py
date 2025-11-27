#! /usr/bin/env python3
from .fix_logging import fix_logging
from .io import InputFile, listing
from .config import Config
from .types import TakeoutDirectoryType, DatabaseFileType
from .db import BatchInserter
from . import sql
from .classifier import classify

import concurrent.futures as cf
import duckdb
import face_recognition
import io
import json
import logging
import moviepy
import numpy as np
import os
import re
import time

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from datetime import datetime
from itertools import repeat
from typing import (
    Optional,
    Tuple,
    Iterable,
    TypedDict,
    Required,
    Any,
    cast,
)
from pathlib import Path
import typer
from rich.progress import track


logger = logging.getLogger("load")

YYYY = r"([12][901][0-9][0-9])"
MM = r"(0[1-9]|1[012])"
DD = r"(0[1-9]|[12][0-9]|3[01])"
YYYYMMDD = YYYY + MM + DD
YYYYMMDD_RE = re.compile(YYYYMMDD)
HHMMSS = r"([0-2][0-9])([0-6][0-9])([0-6][0-9])"
YYYYMMDD_HHMMSS_RE = re.compile(f"{YYYYMMDD}_{HHMMSS}")
YYYY_MM_DD_RE = re.compile(f"{YYYY}[-/]{MM}[-/]{DD}")
NOT_DIGITS = r"[^0-9]"
ISOLATED_YEAR_RE = re.compile(NOT_DIGITS + YYYY + NOT_DIGITS)

START_OF_TIME = datetime(1990, 1, 1, 0, 0, 0)
IMAGE_SIZE_TOO_SMALL = 150
ITHUMBNAIL = (200, 200)
VTHUMBNAIL = 200

STOP_WORDS = {
    "and",
    "folder",
    "for",
    "from",
    "goes",
    "in",
    "is",
    "of",
    "photo",
    "photos",
    "to",
    "trip",
    "untitled",
    'before',
    'google',
    'image',
    'jpg',
    'png',
    'over'
    'the',
    'video',
    'videos',
    'with',
}


class FaceEncoding:
    """vector describing a face and its unique id"""

    def __init__(self, face_id: int, encoding: np.ndarray):
        self.id = face_id
        self.encoding = encoding


class FacesRecord:
    """result of searching an image for faces"""

    def __init__(self):
        self.matches: list[FaceEncoding] = []
        self.new_faces: list[tuple[np.ndarray, bytes]] = []

    def add_face_match(self, encoding: FaceEncoding) -> None:
        self.matches.append(encoding)

    def add_new_face(self, encoding: np.ndarray, thumbnail: bytes) -> None:
        self.new_faces.append((encoding, thumbnail))

NO_FACES = FacesRecord()

class ImageRecord(TypedDict, total=False):
    """try to build up an image description with these bits:"""

    path: Required[str]
    archive: str
    mimetype: str
    hash: str
    size: int
    width: int
    height: int
    taken: datetime
    lat: float
    lon: float
    thumbnail: bytes
    words: list[str]
    faces: FacesRecord

IMAGE_RECORD_TABLE_COLUMNS: list[str] = list(ImageRecord.__mutable_keys__ - {"faces"})

def date_from_filename(filename: str) -> Optional[datetime]:
    """Look for reasonable patterns in filenames and treat them as a datetime"""

    # do not attempt to get a date from the takeout files
    last = filename.split("/")[-1]
    if last.endswith('.zip') and last.startswith('takeout-'):
        return None
    for pattern in [YYYYMMDD_HHMMSS_RE, YYYYMMDD_RE, YYYY_MM_DD_RE, ISOLATED_YEAR_RE]:
        m = pattern.search(filename)
        if m:
            result: Optional[datetime] = None
            try:
                if len(m.groups()) == 6:
                    yyyy = int(m.group(1))
                    mm = int(m.group(2))
                    dd = int(m.group(3))
                    hh = int(m.group(4))
                    mn = int(m.group(5))
                    ss = int(m.group(6))
                    result = datetime(yyyy, mm, dd, hh, mn, ss)
                elif len(m.groups()) == 3:
                    yyyy = int(m.group(1))
                    mm = int(m.group(2))
                    dd = int(m.group(3))
                    result = datetime(yyyy, mm, dd, 0, 0, 0)
                elif len(m.groups()) == 1:
                    yyyy = int(m.group(1))
                    mm = 1
                    dd = 1
                    result = datetime(yyyy, mm, dd, 0, 0, 0)
            except ValueError as e:
                logger.debug("Skipping bad datetime processing with %s: %s", filename, e)

            if result and datetime.now() > result > START_OF_TIME:
                return result

    return None

PUNTUATION = re.compile(r"[\ \(\)\-_\.,'!\"]")
def words_of_filename(filename: str) -> Optional[list[str]]:
    """pull descriptive words from the filename"""
    # descriptive text means "has a space in it"
    if filename.find(" "):
        filename = filename.lower()

        words = set()
        for part in filename.split(os.path.sep):
            if part.find(' ') < 0:
                continue
            for word in PUNTUATION.split(part):
                if len(word) < 2 or word[0].isnumeric():
                    continue
                words.add(word)
        words -= STOP_WORDS
        return sorted(words)
    return None


def commas(seq: Iterable[str]):
    return ",".join(seq)


def fetch_known_faces(db: duckdb.DuckDBPyConnection) -> list[FaceEncoding]:
    result = []
    for face_id, encoding in sql.KNOWN_FACES.fetchmany(db):
        result.append(FaceEncoding(face_id, np.array(encoding)))
    return result


def add_file(
        path: str, archive: str, images: BatchInserter, metadata: BatchInserter
) -> None:
    # if Google can't decode it, don't even try
    if path.find("/Failed Videos/") >= 0:
        logger.warning("Skipping %s", path)
        return

    input_file = InputFile(path, archive)
    mimetype = input_file.mimetype()

    # skip unknown types
    if mimetype == InputFile.UNKNOWN_TYPE:
        return

    # generated by Google, but don't know what to do with it, yet
    if path.endswith("/metadata.json"):
        return

    # supplemental metadata?
    if mimetype in {"application/json"}:
        image_path = path.rsplit(".", 2)[0]
        metadata.add_row(image_path, path, archive)
        return

    # skip types we don't care about
    maintype = mimetype.split("/")[0]
    if maintype not in {"image", "video"}:
        logger.info("Skipping %s of type %s", InputFile(path, archive), mimetype)
        return

    # insert filename/archive
    images.add_row(path, archive)


def best_time(meta: dict[str, Any]) -> Optional[datetime]:
    'pick the best time from "creationTime" and "photoTakenTime"'
    times = []
    for key in "photoTakenTime", "creationTime":
        if key in meta:
            ts = int(meta[key]["timestamp"])
            dt = datetime.fromtimestamp(ts)
            # Occasionally the time is just after the time epoch.
            # Ignore it if it not "reasonable".
            if dt > START_OF_TIME:
                times.append(dt)
    if len(times) == 1:
        return times[0]
    # return the latest
    if len(times) > 1:
        return sorted(times)[0]
    return None


def process_metadata(ir: ImageRecord, metafile: InputFile) -> ImageRecord:
    """fetch metadata from the json file extracted by google from the given image file"""

    taken: Optional[datetime] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    try:
        with metafile.open() as fp:
            meta = json.load(fp)
            taken = best_time(meta)
            lat = meta["geoData"]["latitude"]
            lon = meta["geoData"]["longitude"]
            if lat == 0.0 and lon == 0.0:
                lat = None
                lon = None
    except FileNotFoundError:
        logger.debug("Could not find metadata file %s", metafile)
    if taken:
        ir.update(taken=taken)
    if lat is not None and lon is not None:
        ir.update(lat=lat, lon=lon)
    return ir


def is_small(coords: tuple[int, int, int, int]) -> bool:
    top, right, bottom, left = coords
    return (
        abs(top - bottom) < IMAGE_SIZE_TOO_SMALL
        or abs(left - right) < IMAGE_SIZE_TOO_SMALL
    )

def thumbnail(im: Image.Image) -> bytes:
    try:
        with io.BytesIO() as buf:
            im.thumbnail(size=ITHUMBNAIL)
            im = im.convert("RGB")  # remove any transparency
            im.save(buf, format="jpeg")
            buf.seek(io.SEEK_SET, 0)
            return buf.read()
    except Exception as e:
        logger.exception("Error generating thumbnail for face: %s", e)
    return b''

def scan_faces(input_file: InputFile, all_faces: list[FaceEncoding]) -> FacesRecord:
    result = FacesRecord()
    with input_file.open() as fp:
        image = face_recognition.load_image_file(fp)
    locations: list[tuple[int, int, int, int]] = face_recognition.face_locations(image)
    if locations:
        # quick check: if all the faces are small, don't bother
        # computing the encodings
        if all(map(is_small, locations)):
            return result

        encodings = face_recognition.face_encodings(
            image, known_face_locations=locations
        )

        # sanity check: number of faces found equals the number of encodings
        if len(encodings) != len(locations):
            logger.info(
                "Found %d encodings and %d in %s ", len(encodings), len(locations), input_file
            )
            return result

        # for each location, find the face in the known list of faces
        for location, encoding in zip(locations, encodings):
            if is_small(location):
                continue
            matches = face_recognition.compare_faces(
                [f.encoding for f in all_faces], encoding
            )
            for face, match in zip(all_faces, matches):
                if match:
                    logger.info("Found a match on the %dth face", face.id)
                    result.add_face_match(face)
                    break
            else:
                logger.info("Found a new face")
                top, right, bottom, left = location
                pil_image = Image.fromarray(image[top:bottom, left:right])
                result.add_new_face(encoding, thumbnail(pil_image))

    return result


def process_image_file(
        input_file: InputFile, config: Config, all_faces: list[FaceEncoding]
) -> ImageRecord:
    logger.info("Processing image file %s", input_file)

    words : list[str] = []
    record = ImageRecord(path=input_file.path, archive=input_file.archive)
    digest, size = input_file.hash_and_size()
    record.update(hash=digest, size=size)
    record.update(mimetype=input_file.mimetype())

    with input_file.open() as fp:
        try:
            with Image.open(fp) as im:
                # rotate image to match orientation from the exif data
                ImageOps.exif_transpose(im, in_place=True)
                width, height = im.size
                words += classify(im)
                record.update(width=width, height=height)
                record.update(thumbnail=thumbnail(im))
        except Exception as ex:
            logger.exception("Unable to read image %s: %s", input_file, ex)
            return record

    if config.scan_faces:
        record.update(faces=scan_faces(input_file, all_faces))
    taken = date_from_filename(input_file.path) or date_from_filename(input_file.archive)
    if taken:
        record.update(taken=taken)
<<<<<<< HEAD
    words += words_of_filename(tf.path)
=======
    words = words_of_filename(input_file.path)
>>>>>>> refs/remotes/origin/main
    if words:
        record.update(words=sorted(words))
    return record


def process_video_file(input_file: InputFile, config: Config) -> ImageRecord:
    logger.info("Processing video file %s", input_file)

    record = ImageRecord(path=input_file.path, archive=input_file.archive)

    digest, size = input_file.hash_and_size()
    record.update(hash=digest, size=size)
    record.update(mimetype=input_file.mimetype())

    taken = date_from_filename(input_file.path)
    if taken:
        record.update(taken=taken)
    words = words_of_filename(input_file.path)
    if words:
        record.update(words=words)

    try:
        with input_file.asfile() as simple_file:
            with moviepy.VideoFileClip(simple_file.filename()) as vc:
                width, height = vc.size
                record.update(width=width, height=height)

                # generate a gif from the first few seconds of the video

                # I struggled with moviepy type-annotations
                # pyrefly: ignore  # not-callable, bad-assignment, missing-argument
                vc = vc.without_audio()
                # pyrefly: ignore  # not-callable, bad-assignment
                vc = vc.with_end(5)
                vc = vc.with_effects([moviepy.vfx.Resize(width=VTHUMBNAIL)])
                filename1 = simple_file.filename()
                stem, _ = os.path.splitext(filename1)
                gif = stem + ".gif"
                vc.write_gif(gif, fps=1, loop=0, logger=None)  # loop=0 means loop forever
                try:
                    with open(gif, "rb") as fp:
                        record.update(thumbnail=fp.read())
                finally:
                    os.remove(gif)
    except Exception as e:
        logger.exception("Error processing video file %s: %s", input_file, e)
    return record


def process_file(paths: Tuple[str, str, str, str], config: Config, all_faces: list[FaceEncoding]) -> Optional[ImageRecord]:
    fix_logging(config.log)
    path, archive, meta_path, meta_archive = paths
    input_file = InputFile(path, archive)
    mimetype = input_file.mimetype()

    # unknown type, skip
    if mimetype == InputFile.UNKNOWN_TYPE:
        return None

    mimetype = mimetype.split("/")[0]

    ir : Optional[ImageRecord] = None
    if mimetype == "video":
        ir = process_video_file(input_file, config)

    elif mimetype == "image":
        ir = process_image_file(input_file, config, all_faces)

    else:
        logger.error("Unknown mimetype %s for %s", mimetype, path)

    if ir and meta_path:
        ir = process_metadata(ir, InputFile(meta_path, meta_archive))

    return ir


class Loader:
    
    def __init__(self, config: Config):
        self.config = config
        self.db = duckdb.connect(self.config.database)

    def close(self):
        self.db.close()

    def create_tables(self):
        with self.db.cursor() as cur:
            sql.create_tables(cur)

    def scan_files(self) -> None:
        start = self.config.load_directory
        logger.info("Scanning files in %s", start)

        with self.db.cursor() as cur, \
             BatchInserter(cur, "image_files", ["path", "archive"]) as images, \
             BatchInserter(cur, "meta_files", ["image_path", "meta_path", "archive"]) as metadata:

            # walk the tree and find all images/video
            for dirpath, dir_names, filenames in os.walk(start, followlinks=True):

                # skip the Failed Videos directory
                dir_names[:] = [d for d in dir_names if d != "Failed Videos"]
                for filename in filenames:
                    path = os.path.join(dirpath, filename)

                    if filename.endswith(".zip"):
                        archive = path
                        for path in listing(archive):
                            add_file(path, archive, images, metadata)
                    else:
                        add_file(path, "", images, metadata)


    def load_files(self) -> int:
        
        # We can't process database updates in the process pool
        # because the database connection cannot be shared in worker
        # processes. So prepare ImageRecords in parallel and batch
        # insert them in the main thread.
        
        with self.db.cursor() as cur, \
             BatchInserter(cur, "images", IMAGE_RECORD_TABLE_COLUMNS, max=5) as images_inserter, \
             BatchInserter(cur, "faces", ["id", "encoding", "image"]) as faces_inserter, \
             BatchInserter(cur, "face_matches", ["path", "face_id"]) as matches_inserter, \
             cf.ProcessPoolExecutor(max_workers=self.config.concurrency) as executor:

                all_faces = fetch_known_faces(cur)
                next_id = max([f.id for f in all_faces], default=-1) + 1
                row = sql.IMAGE_TOTAL_PROCESSED_COUNTS.fetchone(cur)
                total, processed = cast(Tuple[int, int], row)
                and_faces = " and faces" if self.config.scan_faces else ""
                logger.info("Processing %d images for thumbnails%s", total - processed, and_faces)

                # The following query loads all the outstanding image
                # files into memory, which is not ideal.  However,
                # long-running queries can cause deadlock when the
                # duckdb WAL is commited as we write updates.  See
                # https://github.com/duckdb/duckdb/issues/14303
                
                paths = sql.WORK_LIST.fetchall(cur)
                for ir in track(
                        executor.map(
                            process_file,
                            paths,
                            repeat(self.config),
                            repeat(all_faces),
                            buffersize=self.config.concurrency),
                        completed=processed,
                        total=total):
                    if ir:
                        images_inserter.add_dict(ir)
                        faces : FacesRecord = ir.get("faces", NO_FACES)
                        for face in faces.matches:
                            matches_inserter.add_row(ir["path"], face.id)
                        for encoding, thumbnail in faces.new_faces:
                            enc = FaceEncoding(next_id, encoding)
                            next_id += 1
                            all_faces.append(enc)
                            faces_inserter.add_row(enc.id, enc.encoding, thumbnail)
                            matches_inserter.add_row(ir["path"], enc.id)

                return total - processed


def start(
    load_directory: TakeoutDirectoryType,
    database: DatabaseFileType = Path("images.db"),
    concurrency: int = 0,
    skip_faces: bool = False,
    log: Path = Path("load.log"),
) -> None:
    # add support for HEIF image types
    register_heif_opener()

    config = Config(load_directory, database, concurrency, not skip_faces, log)
    fix_logging(config.log)

    loader = Loader(config)

    start = time.time()
    loader.create_tables()
    loader.scan_files()
    processed = loader.load_files()
    seconds = time.time() - start

    with loader.db.cursor() as cur:
        thumbnails = sql.THUMBNAIL_COUNT.count(cur)
        faces = sql.FACE_COUNT.count(cur)
        matches = sql.FACE_MATCH_COUNT.count(cur)
        matched_photos = sql.IMAGES_WITH_FACES_COUNT.count(cur)
        dups = sql.IMAGE_DUPLICATE_COUNT.count(cur)
        
    logger.info("Processed %d files in %.1f seconds", processed, seconds)
    logger.info(
        "Found %d faces in %d images (%d total matches)", faces, matched_photos, matches
    )
    logger.info("There are %d duplicate images", dups)
    logger.info("There are %d total thumbnails", thumbnails)


def main():
    typer.run(start)


if __name__ == "__main__":
    main()
