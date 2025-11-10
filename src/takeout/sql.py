from typing import Tuple, Any
from .db import fetch_many, count
import duckdb
import re

def create_tables(cur: duckdb.DuckDBPyConnection) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
               path VARCHAR NOT NULL PRIMARY KEY,
               archive VARCHAR,
               mimetype VARCHAR,
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

class QueryException(Exception):
    pass

class Query:
    def __init__(self, q: str, binds: list[Tuple[str, type]] = []):
        self.q = q
        self.binds = binds[:]
        parts = re.findall('[?]', q)
        assert len(parts) == len(binds), \
            f"unmatched number of bind locations ({len(parts)}) " \
            f"with bind description {[name for name, type in binds]}: " \
            f"{self.q}"

    def check_bind(self, binds: list[Any]) -> None:
        if len(binds) != len(self.binds):
            raise QueryException(f"binding length ({len(self.binds)}) does not match required binds (self.binds)")
        for i, ((name, atype), value) in enumerate(zip(self.binds, binds)):
            if not isinstance(value, atype):
                raise QueryException(f"Value {value} for binding {name} (argument {i}) is not a {atype}")

    def fetchone(self, cur: duckdb.DuckDBPyConnection, binds: list[Any] = []):
        self.check_bind(binds)
        return cur.execute(self.q, binds).fetchone()

    def fetchall(self, cur: duckdb.DuckDBPyConnection, binds: list[Any] = []):
        self.check_bind(binds)
        return cur.execute(self.q, binds).fetchall()

    def fetchmany(self, cur: duckdb.DuckDBPyConnection, binds: list[Any] = []):
        self.check_bind(binds)
        return fetch_many(cur, self.q, binds)

    def count(self, cur: duckdb.DuckDBPyConnection, default=0):
        self.check_bind([])
        return count(cur, self.q, default)

WORK_LIST = Query(
    """SELECT if.path, if.archive, m.meta_path, m.archive
         FROM image_files if
         LEFT JOIN meta_files m ON if.path = m.image_path
         LEFT JOIN images i ON if.path = i.path
       WHERE i.hash IS NULL
    """)

THUMBNAIL_COUNT = Query(
    """SELECT COUNT(*)
         FROM images
        WHERE thumbnail IS NOT NULL
    """)

FACE_COUNT = Query("SELECT COUNT(*) FROM faces")

FACE_MATCH_COUNT = Query("SELECT COUNT(*) FROM face_matches")

IMAGES_WITH_FACES_COUNT = Query(
    """SELECT COUNT(*) FROM (
           SELECT path FROM face_matches GROUP BY path
       )
    """)

IMAGE_DUPLICATE_COUNT = Query(
    """SELECT COUNT(*) FROM (
          SELECT count(hash) c FROM images GROUP BY hash
       )
       WHERE c > 1
    """)

KNOWN_FACES = Query("SELECT id, encoding FROM faces")

IMAGE_TOTAL_PROCESSED_COUNTS = Query(
    """SELECT count(if.path), count(i.hash)
         FROM image_files if
         LEFT JOIN images i ON if.path = i.path
    """)

LOCATION_COMPLETION_QUERY = Query(
    """SELECT display_name
         FROM places
        WHERE display_name ILIKE CONCAT(?, '%')
        ORDER BY population DESC
        LIMIT ?
    """,
    [('prefix', str),
     ('limit', int)])

FACE_QUERY = Query(
    """SELECT face_id, c
         FROM (
              SELECT face_id, count(*) c
              FROM face_matches
              GROUP BY face_id
         )
        ORDER BY c DESC
        LIMIT 40
    """)

THUMBNAIL_QUERY = Query(
    """SELECT thumbnail
         FROM images
        WHERE hash = ?
    """, [('hash', str)])

IMAGE_DETAIL = Query(    
    """SELECT path,
              archive,
              mimetype,
              size,
              width,
              height,
              taken,
              words,
              lat,
              lon
         FROM images
        WHERE hash = ?
    """,[("hash", str)])

NEAREST_LOCATION = Query(
    """INSTALL spatial;
       LOAD spatial;
       SELECT display_name,
              ST_Distance_Spheroid(ST_Point(lat, lon), ST_Point(?, ?)) distance
         FROM places
        WHERE distance < ?
        ORDER by distance ASC
        LIMIT 1
    """, [
        ('lat', float),
        ('lon', float),
        ('distance', float)
    ])

FACES_FOR_IMAGE = Query(
    """SELECT face_id
         FROM face_matches f JOIN images i on f.path = i.path
        WHERE i.hash = ?
        GROUP BY face_id
    """, [('hash', str)])

IMAGE_INPUT = Query(
    """SELECT path, archive FROM images WHERE hash = ?
    """, [('hash', str)])

FETCH_FACE = Query(
    """SELECT image FROM faces WHERE id = ?""",
    [("id", int)])

YEARS = Query(
    """SELECT year(taken) y
         FROM images
        GROUP BY y
        ORDER BY y desc
    """)
