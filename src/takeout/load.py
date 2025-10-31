#! /usr/bin/env python3
from .fix_logging import fix_logging
from .io import TakeoutFile, listing
from .config import Config
from .types import TakeoutDirectoryType, DatabaseFileType
from .db import fetch_many, count

import concurrent.futures as cf
import duckdb
import face_recognition
import io
import json
import logging
import moviepy
import numpy as np
import os
import pandas as pd
import re
import time

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from datetime import datetime
from itertools import repeat
from typing import (
    Optional,
    Any,
    Tuple,
    Iterable,
    TypedDict,
    Required,
)
from pathlib import Path
import typer

fix_logging()

logger = logging.getLogger("load")

YYYY = r"([12][901][0-9][0-9])"
MM = r"([01][0-9])"
DD = r"([0123][0-9])"
YYYYMMDD = YYYY + MM + DD
YYYYMMDD_RE = re.compile(YYYYMMDD)
HHMMSS = r"([0-2][0-9])([0-6][0-9])([0-6][0-9])"
YYYYMMDD_HHMMSS_RE = re.compile(f"{YYYYMMDD}_{HHMMSS}")
YYYY_MM_DD_RE = re.compile(f"{YYYY}-{MM}-{DD}")
ISOLATED_YEAR_RE = re.compile(r"[^0-9]" + YYYY + "[^0-9]")

START_OF_TIME = datetime(1990, 1, 1, 0, 0, 0)
IMAGE_SIZE_TOO_SMALL = 150

STOP_WORDS = {
    "1",
    "to",
    "goes",
    "trip",
    "favorites",
    "for",
    "untitled",
    "from",
    "photos",
    "photo",
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
            if result and datetime.now() > result > START_OF_TIME:
                return result

    return None


def words_of_filename(filename: str) -> Optional[list[str]]:
    """extract anything interesting from the album name"""
    # name of directory containing the file
    parent, file = os.path.split(filename)
    base, parent = os.path.split(parent)
    # ignore if there are no spaces in the filename
    if parent.find(" "):
        parent = parent.lower()
        # remove punctuation
        for c in "()-":
            parent = parent.replace(c, " ")
        parts = set(parent.split()) - STOP_WORDS
        return list(parts)
    return None


def process_metadata(metadata_args=Tuple[str, str, str]) -> ImageRecord:
    """fetch metadata from the json file extracted by google from the given image file"""

    image_path, meta_path, meta_archive = metadata_args

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
    if taken is None:
        taken = date_from_filename(image_path)
    words = words_of_filename(image_path)
    result = ImageRecord(path=image_path)
    if taken:
        result.update(taken=taken)
    if lat is not None and lon is not None:
        result.update(lat=lat, lon=lon)
    if words:
        result.setdefault("words", []).extend(words)
    return result


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

        for location, encoding in zip(locations, encodings):
            if is_small(location):
                continue
            matches = face_recognition.compare_faces(
                [f.encoding for f in all_faces], encoding
            )
            for j, match in enumerate(matches):
                face = all_faces[j]
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
                        pil_image.thumbnail(size=(120, 120))
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
    filename: str, archive: str, all_faces: list[FaceEncoding]
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
    record.update(faces=scan_faces(tf, all_faces))

    return record


def process_video_file(filename: str, archive: str) -> ImageRecord:
    logger.info("Processing video file %s", filename)

    record = ImageRecord(path=filename, archive=archive)
    tf = TakeoutFile(filename, archive)

    digest, size = tf.hash_and_size()
    record.update(hash=digest, size=size)

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
            vc = vc.with_effects([moviepy.vfx.Resize(width=128)])
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


def commas(seq: Iterable[str]):
    return ",".join(seq)


def process_results(
    config: Config, all_faces: list[FaceEncoding], record: Optional[ImageRecord]
) -> None:
    if record is None:
        return None
    with config.cursor() as cur:
        column_names = list(record.keys() - {"faces"})
        # generate an insert statement that does an update
        # on conflict
        try:
            binding_string = commas(["?" for _ in column_names])
            binding = [record.get(k) for k in column_names]
            insert = f"""
               INSERT INTO images({commas(column_names)})
               VALUES ({binding_string})
            """
            insert += "ON CONFLICT DO UPDATE SET "
            insert += commas(
                [f"{c} = EXCLUDED.{c}" for c in record.keys() - {"path", "faces"}]
            )
            cur.execute(insert, binding)
        except Exception as e:
            logger.exception("Unable to insert records: %s", e)
            return  # noqa: F841

        if "faces" in record:
            path = record["path"]
            for face in record["faces"].matches:
                cur.execute(
                    """
                    INSERT INTO face_matches(path, face_id) VALUES (?, ?)
                    """,
                    [path, face.id],
                )
            next_id = max([f.id for f in all_faces], default=-1) + 1
            for encoding, thumbnail in record["faces"].new_faces:
                enc = FaceEncoding(next_id, encoding)
                next_id = next_id + 1
                all_faces.append(enc)
                cur.execute(
                    """
                    INSERT INTO faces(id, encoding, image) VALUES(?, ?, ?)
                    """,
                    [enc.id, enc.encoding, thumbnail],
                )


def fetch_known_faces(db: duckdb.DuckDBPyConnection) -> list[FaceEncoding]:
    result = []
    for face_id, encoding in fetch_many(db, "SELECT id, encoding FROM faces"):
        result.append(FaceEncoding(face_id, np.array(encoding)))
    return result


class BatchInserter:
    def __init__(self, config: Config, table: str, columns: list[str], max: int = 100):
        self.config = config
        self.table = table
        self.columns = columns
        self.data: dict[str, list[Any]] = {c: [] for c in self.columns}
        self.count = 0
        self.max = max

    def flush(self) -> None:
        if self.data:
            col_str = ", ".join(self.columns)
            df = pd.DataFrame(self.data)  # noqa: F841
            with self.config.cursor() as cur:
                stmt = f"""
                      INSERT OR IGNORE
                        INTO {self.table} ({col_str})
                      SELECT * from df
                    """
                cur.execute(stmt)
            self.data = {c: [] for c in self.columns}
            self.count = 0

    def add_row(self, *cols) -> None:
        for name, value in zip(self.columns, cols):
            self.data[name].append(value)
            self.count += 1
        if self.count > self.max:
            self.flush()


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


def scan_files(config: Config) -> None:
    start = config.takeout_directory
    logger.info("Scanning files in %s", start)

    images = BatchInserter(config, "images", ["path", "archive"])
    metadata = BatchInserter(
        config, "metafiles", ["image_path", "meta_path", "archive"]
    )

    # walk the tree and find all images/video
    for dirpath, dir_names, filenames in os.walk(start):
        
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

    with config.cursor() as db:
        total = count(db, "SELECT count(*) FROM images")
        unprocessed = count(db, "SELECT count(*) FROM images WHERE hash IS NULL")
        incomplete = total - unprocessed
        logger.info(
            "There are %d known files, %d unprocessed (%1.f%%)",
            total,
            unprocessed,
            incomplete * 100.0 / total if total > 0 else 0.,
        )


def process_file(file_reference: Tuple[str, str], all_faces) -> Optional[ImageRecord]:
    path, archive = file_reference
    tf = TakeoutFile(path, archive)
    mimetype = tf.mimetype()

    # unknown type, skip
    if mimetype == TakeoutFile.UNKNOWN_TYPE:
        return None

    mimetype = mimetype.split("/")[0]

    if mimetype == "video":
        return process_video_file(path, archive)

    elif mimetype == "image":
        return process_image_file(path, archive, all_faces)

    else:
        logger.error("Unknown mimetype %s for %s", mimetype, path)

    return None


def load_files(config: Config) -> None:
    with config.cursor() as db:
        # all_faces will be updated as we process images
        all_faces = fetch_known_faces(db)

        # We can't process database updates in the process pool
        # because the database connection cannot be shared in worker
        # processes. So prepare an ImageRecord in parallel to be added
        # in the main thread.

        with cf.ProcessPoolExecutor() as executor:
            logger.info("Loading image thumbnails, finding faces")

            # call process_file( (path, archive), all_faces ) for each entry without a hash
            path_archive = fetch_many(
                # unprocessed files will have a null hash
                db,
                """SELECT path, archive FROM images WHERE hash IS NULL""",
            )
            for ir in executor.map(
                process_file,
                path_archive,
                repeat(all_faces),
                buffersize=config.concurrency,
            ):
                process_results(config, all_faces, ir)

            # extract metadata from the metadata files
            logger.info("Loading metadata")
            metadata = fetch_many(
                db,
                """
                   SELECT m.image_path, m.meta_path, m.archive
                     FROM metafiles m, images i
                    WHERE i.path = m.image_path
                      AND i.taken is null
                    """,
            )
            for ir in executor.map(
                process_metadata, metadata, buffersize=config.concurrency
            ):
                process_results(config, all_faces, ir)


def create_tables(config: Config):
    with config.cursor() as db:
        db.execute(
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
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS metafiles (
                   image_path VARCHAR NOT NULL PRIMARY KEY,
                   meta_path VARCHAR NOT NULL,
                   archive VARCHAR NOT NULL
            );
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS faces (
                   id INTEGER NOT NULL PRIMARY KEY,
                   encoding DOUBLE[] NOT NULL,
                   image BLOB
            );
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS face_matches (
                   path VARCHAR NOT NULL,
                   face_id INTEGER NOT NULL
            );
            """
        )

def start(
    takeout_directory: TakeoutDirectoryType,
    database: DatabaseFileType = Path("images.db"),
    concurrency: int = 0,
) -> None:
    # add support for HEIF image types
    register_heif_opener()

    config = Config(takeout_directory, database, concurrency)

    now = time.time()
    create_tables(config)
    scan_files(config)
    load_files(config)
    seconds = time.time() - now

    with config.cursor() as cur:
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
        
    logger.info("Stored %d thumbnails in %.1f seconds", thumbnails, seconds)
    logger.info(
        "Found %d faces in %d images (%d total matches)", faces, matched_photos, matches
    )
    logger.info("There are %d duplicate images", dups)


def main():
    typer.run(start)


if __name__ == "__main__":
    typer.run(start)
