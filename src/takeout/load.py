#! /usr/bin/env python3
from .fix_logging import fix_logging
from .io import TakeoutFile, listing
from .config import Config
from .types import TakeoutDirectoryType, DatabaseFileType
from .db import fetch_many, count, BatchInserter

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
)
from pathlib import Path
import typer
from rich.progress import track


logger = logging.getLogger("load")

YYYY = r"([12][901][0-9][0-9])"
MM = r"([01][0-9])"
DD = r"([0123][0-9])"
YYYYMMDD = YYYY + MM + DD
YYYYMMDD_RE = re.compile(YYYYMMDD)
HHMMSS = r"([0-2][0-9])([0-6][0-9])([0-6][0-9])"
YYYYMMDD_HHMMSS_RE = re.compile(f"{YYYYMMDD}_{HHMMSS}")
YYYY_MM_DD_RE = re.compile(f"{YYYY}[-/]{MM}[-/]{DD}")
ISOLATED_YEAR_RE = re.compile(r"[^0-9]" + YYYY + "[^0-9]")

START_OF_TIME = datetime(1990, 1, 1, 0, 0, 0)
IMAGE_SIZE_TOO_SMALL = 150
ITHUMBNAIL = (200, 200)
VTHUMBNAIL = 200

STOP_WORDS = {
    "to",
    "in",
    "of",
    "goes",
    "trip",
    "favorites",
    "for",
    "untitled",
    "from",
    "folder",
    "photos",
    "photo",
    'image',
    'jpg',
    'the',
    'family',
    'before',
    '2000',
    'over'
    'with',
    'google',
    'video',
    'videos',
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

def date_from_filename(filename: str) -> Optional[datetime]:
    """Look for reasonable patterns in filenames and treat them as a datetime"""
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
                logger.debug(f"Skipping bad datetime processing with {filename}: {e}")

            if result and datetime.now() > result > START_OF_TIME:
                return result

    return None


def words_of_filename(filename: str) -> Optional[list[str]]:
    """extract anything interesting from the album name"""
    # ignore if there are no spaces in the filename
    if filename.find(" "):
        filename = filename.lower()
        space_parts = [part for part in filename.split("/") if part.find(' ') >= 0]
        # remove punctuation
        parts = set()
        for part in space_parts:
            for c in "()-.,'":
                part = part.replace(c, " ")
            parts.update([w for w in part.split() if len(w) > 1])
        parts -= STOP_WORDS
        return sorted(list(parts))
    return None


def commas(seq: Iterable[str]):
    return ",".join(seq)


def fetch_known_faces(db: duckdb.DuckDBPyConnection) -> list[FaceEncoding]:
    result = []
    for face_id, encoding in fetch_many(db, "SELECT id, encoding FROM faces"):
        result.append(FaceEncoding(face_id, np.array(encoding)))
    return result


def add_file(
    images: BatchInserter, metadata: BatchInserter, path: str, archive: str
) -> None:
    # if Google can't decode it, don't even try
    if path.find("/Failed Videos/") >= 0:
        logger.warning("Skipping %s", path)
        return

    tf = TakeoutFile(path, archive)
    mimetype = tf.mimetype()

    # skip unknown types
    if mimetype == TakeoutFile.UNKNOWN_TYPE:
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
        logger.info("Skipping %s of type %s", TakeoutFile(path, archive), mimetype)
        return

    # insert filename/archive
    images.add_row(path, archive)


def process_metadata(ir: ImageRecord, meta_path: str, meta_archive: str) -> ImageRecord:
    """fetch metadata from the json file extracted by google from the given image file"""

    taken: Optional[datetime] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    tf = TakeoutFile(meta_path, meta_archive)
    try:
        with tf.open() as fp:
            meta = json.load(fp)
            ts = int(meta["photoTakenTime"]["timestamp"])
            taken = datetime.fromtimestamp(ts)
            lat = meta["geoData"]["latitude"]
            lon = meta["geoData"]["longitude"]
            if lat == 0.0 and lon == 0.0:
                lat = None
                lon = None
    except FileNotFoundError:
        logger.debug("Could not find metadata file %s", tf)
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


def scan_faces(tf: TakeoutFile, all_faces: list[FaceEncoding]) -> FacesRecord:
    result = FacesRecord()
    with tf.open() as fp:
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
                "Found %d encodings and %d in %s ", len(encodings), len(locations), tf
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
                try:
                    with io.BytesIO() as buf:
                        pil_image.thumbnail(size=ITHUMBNAIL)
                        pil_image = pil_image.convert("RGB")  # remove any transparency
                        pil_image.save(buf, format="jpeg")
                        buf.seek(io.SEEK_SET, 0)
                        thumbnail = buf.read()
                except Exception as e:
                    logger.exception("Error generating thumbnail for face: %s", e)
                    thumbnail = b""
                result.add_new_face(encoding, thumbnail)

    return result


def process_image_file(
        filename: str, archive: str, config: Config, all_faces: list[FaceEncoding]
) -> ImageRecord:
    logger.info("Processing image file %s", filename)

    record = ImageRecord(path=filename, archive=archive)
    tf = TakeoutFile(filename, archive)
    digest, size = tf.hash_and_size()
    record.update(hash=digest, size=size)

    with tf.open() as fp:
        try:
            with Image.open(fp) as im:
                # rotate image to match orientation from the exif data
                ImageOps.exif_transpose(im, in_place=True)
                width, height = im.size
                record.update(width=width, height=height)

                # generate thumbnail
                with io.BytesIO() as buf:
                    im.thumbnail((120, 120))
                    im = im.convert("RGB")  # remove any transparency
                    im.save(buf, format="jpeg")
                    buf.seek(io.SEEK_SET, 0)
                    thumbnail = buf.read()
        except Exception as ex:
            logger.exception("Unable to read image %s: %s", tf, ex)
            return record

    record.update(thumbnail=thumbnail)
    if config.scan_faces:
        record.update(faces=scan_faces(tf, all_faces))
    taken = date_from_filename(filename) or date_from_filename(archive)
    if taken:
        record.update(taken=taken)
    words = words_of_filename(filename)
    if words:
        record.update(words=words)
    return record


def process_video_file(filename: str, archive: str, config: Config) -> ImageRecord:
    logger.info("Processing video file %s", filename)

    record = ImageRecord(path=filename, archive=archive)
    tf = TakeoutFile(filename, archive)

    digest, size = tf.hash_and_size()
    record.update(hash=digest, size=size)

    taken = date_from_filename(filename)
    if taken:
        record.update(taken=taken)
    words = words_of_filename(filename)
    if words:
        record.update(words=words)

    try:
        with tf.asfile() as simple_file:
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
                return record
    except Exception as e:
        logger.exception(f"Error processing video file {tf}: {e}")


def process_file(paths: Tuple[str, str, str, str], config: Config, all_faces: list[FaceEncoding]) -> Optional[ImageRecord]:
    fix_logging(config.log)
    path, archive, meta_path, meta_archive = paths
    tf = TakeoutFile(path, archive)
    mimetype = tf.mimetype()

    # unknown type, skip
    if mimetype == TakeoutFile.UNKNOWN_TYPE:
        return None

    mimetype = mimetype.split("/")[0]

    ir : Optional[ImageRecord] = None
    if mimetype == "video":
        ir = process_video_file(path, archive, config)

    elif mimetype == "image":
        ir = process_image_file(path, archive, config, all_faces)

    else:
        logger.error("Unknown mimetype %s for %s", mimetype, path)

    if ir and meta_path:
        ir = process_metadata(ir, meta_path, meta_archive)

    return ir


class Loader:
    
    def __init__(self, config: Config):
        self.config = config
        self.db = duckdb.connect(self.config.database)

    def close(self):
        self.db.close()

    def create_tables(self):
        with self.db.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS images (
                       path VARCHAR NOT NULL PRIMARY KEY,
                       archive VARCHAR,
                       hash VARCHAR,
                       size BIGINT,
                       width INTEGER,
                       height INTEGER,
                       taken TIMESTAMP_NS,
                       lat DOUBLE,
                       lon DOUBLE,
                       words VARCHAR[],
                       thumbnail BLOB
                );
                """
            )
            # file loading and image processing are done in two steps:
            #
            # The metadata is stored in separate files (and archives!)
            # so it's possible to see a metadata file and not know
            # what archive the image data is stored in.  So scan all
            # the files first and bulk insert all the names into two
            # work lists: image_files and meta_files.  Then process
            # the work lists and bulk insert all the details into our
            # images table for searching.
            # 
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS images (
                       path VARCHAR NOT NULL PRIMARY KEY,
                       archive VARCHAR,
                       hash VARCHAR,
                       size BIGINT,
                       width INTEGER,
                       height INTEGER,
                       taken TIMESTAMP_NS,
                       lat DOUBLE,
                       lon DOUBLE,
                       words VARCHAR[],
                       thumbnail BLOB
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS image_files (
                       path VARCHAR NOT NULL PRIMARY KEY,
                       archive VARCHAR
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS meta_files (
                       image_path VARCHAR NOT NULL PRIMARY KEY,
                       meta_path VARCHAR NOT NULL,
                       archive VARCHAR NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS faces (
                       id INTEGER NOT NULL PRIMARY KEY,
                       encoding DOUBLE[] NOT NULL,
                       image BLOB
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS face_matches (
                       path VARCHAR NOT NULL,
                       face_id INTEGER NOT NULL,
                CONSTRAINT path_face_id UNIQUE (path, face_id)
                );
                """
            )


    def scan_files(self) -> None:
        start = self.config.takeout_directory
        logger.info("Scanning files in %s", start)

        with self.db.cursor() as cur:
            images = BatchInserter(cur, "image_files", ["path", "archive"])
            metadata = BatchInserter(cur, "meta_files", ["image_path", "meta_path", "archive"])

            # walk the tree and find all images/video
            for dirpath, dir_names, filenames in os.walk(start, followlinks=True):

                # skip the Failed Videos directory
                dir_names[:] = [d for d in dir_names if d != "Failed Videos"]
                for filename in filenames:
                    path = os.path.join(dirpath, filename)

                    if filename.endswith(".zip"):
                        archive = path
                        for path in listing(archive):
                            add_file(images, metadata, path, archive)
                    else:
                        add_file(images, metadata, path, "")

            images.flush()
            metadata.flush()


    def load_files(self) -> int:
        with self.db.cursor() as cur:
            all_faces = fetch_known_faces(cur)
            next_id = max([f.id for f in all_faces], default=-1) + 1

            # We can't process database updates in the process pool
            # because the database connection cannot be shared in
            # worker processes. So prepare ImageRecords in parallel
            # and bulk insert them in the main thread.

            columns : list[str] = list(ImageRecord.__mutable_keys__ - {"faces"})
            images_inserter = BatchInserter(cur, "images", columns)
            faces_inserter = BatchInserter(cur, "faces", ["id", "encoding", "image"])
            matches_inserter = BatchInserter(cur, "face_matches", ["path", "face_id"])

            with cf.ProcessPoolExecutor(max_workers=self.config.concurrency) as executor:
                total = count(cur, "SELECT count(*) from image_files")
                processed = count(cur, """
                   SELECT count(*)
                     FROM image_files if LEFT JOIN images i ON if.path = i.path
                    WHERE i.hash IS NOT NULL
                """)
                and_faces = " and faces" if self.config.scan_faces else ""
                logger.info("Processing %d images for thumbnails%s", total - processed, and_faces)
                
                paths = fetch_many(
                    cur,
                    """SELECT if.path, if.archive, m.meta_path, m.archive
                         FROM image_files if
                              LEFT JOIN meta_files m ON if.path = m.image_path
                              LEFT JOIN images i ON if.path = i.path
                        WHERE i.hash IS NULL
                    """)
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

                for inserter in images_inserter, matches_inserter, faces_inserter:
                    inserter.flush()

                return total - processed


def start(
    takeout_directory: TakeoutDirectoryType,
    database: DatabaseFileType = Path("images.db"),
    concurrency: int = 0,
    skip_faces: bool = False,
    log: Path = Path("load.log"),
) -> None:
    # add support for HEIF image types
    register_heif_opener()

    config = Config(takeout_directory, database, concurrency, not skip_faces, log=log)
    fix_logging(config.log)

    loader = Loader(config)

    now = time.time()
    loader.create_tables()
    loader.scan_files()
    processed = loader.load_files()
    seconds = time.time() - now

    with loader.db.cursor() as cur:
        thumbnails = count(
            cur, "SELECT COUNT(*) FROM images WHERE thumbnail IS NOT NULL"
        )
        faces = count(cur, "SELECT COUNT(*) FROM faces")
        matches = count(cur, "SELECT COUNT(*) FROM face_matches")
        matched_photos = count(
            cur,
            """SELECT COUNT(*) FROM (
                 SELECT path FROM face_matches GROUP BY path
               )
            """
        )
        dups = count(
            cur,
            """SELECT COUNT(*) FROM (
                  SELECT count(hash) c FROM images GROUP BY hash
            ) WHERE c > 1
            """
        )
        
    logger.info("Processed %d files in %.1f seconds", processed, seconds)
    logger.info(
        "Found %d faces in %d images (%d total matches)", faces, matched_photos, matches
    )
    logger.info("There are %d duplicate images", dups)
    logger.info("There are %d total thumbnails", thumbnails)


def main():
    typer.run(start)


if __name__ == "__main__":
    typer.run(start)
