from .fix_logging import fix_logging
import duckdb
import os
import shutil
import io
import pycurl
import logging
import tempfile
import time
import typer
from pathlib import Path
from .types import DatabaseFileType
from .db import count
from zipfile import ZipFile

fix_logging()

logger = logging.getLogger("load_location_data")

# do not overload this source of location data
URL = "https://www.geoapify.com/data-share/localities"


class Headers:
    def __init__(self):
        self.headers = {}

    def decode(self, line_bytes: bytes) -> None:
        line = line_bytes.decode("iso-8859-1")
        if ":" not in line:
            return
        name, value = line.split(":", 1)
        name = name.strip().lower()
        value = value.strip()
        self.headers[name] = value

    def get(self, key, dfault=None):
        return self.headers.get(key.lower(), dfault)


def load_zip_files(database: Path) -> None:
    now = time.time()
    logger.info("initializing database")

    db = duckdb.connect(database)
    db.execute("""
    CREATE OR REPLACE TABLE places (
            name         VARCHAR NOT NULL,
            display_name VARCHAR NOT NULL UNIQUE,
            country_code VARCHAR,
            type         VARCHAR NOT NULL,
            lat          DOUBLE,
            lon          DOUBLE,
            population   INTEGER
    )
    """)

    logger.info("reading files to download")

    buffer = io.BytesIO()
    headers = Headers()
    c = pycurl.Curl()
    c.setopt(pycurl.URL, URL + "/")
    c.setopt(pycurl.WRITEDATA, buffer)
    c.setopt(pycurl.HEADERFUNCTION, headers.decode)
    c.perform()
    c.close()
    html = buffer.getvalue()
    content_type = headers.get("content-type", "text/html")
    if content_type != "text/html":
        raise Exception(f"Unexpected content-type {content_type}")
    zips = []
    for line in html.decode("utf-8").split("\n"):
        quote_start = line.find('"')
        quote_end = line.find('"', quote_start + 1)
        if quote_start >= 0 and quote_end >= 0:
            zip = line[quote_start + 1 : quote_end]
            if zip.find(".zip") > 0:
                zips.append(zip)
    logger.info(f"Found zips: {repr(zips)[:80]}...")

    with tempfile.TemporaryDirectory() as tempdir:
        tmp = os.path.join(tempdir, "tmp.ndjson")
        for zip in zips:
            # be kind to our data source: take a little break between files
            time.sleep(0.25)
            logger.info(f"fetching {zip}")
            zipfile = os.path.join(tempdir, zip)
            with open(zipfile, "wb") as zf:
                headers = Headers()
                c = pycurl.Curl()
                c.setopt(pycurl.URL, URL + "/" + zip)
                c.setopt(pycurl.WRITEDATA, zf)
                c.setopt(pycurl.HEADERFUNCTION, headers.decode)
                c.perform()
                c.close()
            content_type = headers.get("content-type", "text/html")
            if content_type != "application/zip":
                raise Exception(f"Unexpected content-type {content_type}")

            with ZipFile(zipfile) as zf:
                for name in zf.namelist():
                    if name.endswith(".ndjson"):
                        country_code = name.split("/")[0]
                        logger.info(f"loading {name}")
                        # copy out the data into a temp file for loading by duckdb
                        with open(tmp, "wb") as output, zf.open(name, "r") as input:
                            shutil.copyfileobj(input, output)
                        db.execute(f"""
                        INSERT OR REPLACE INTO places(name,
                                           display_name,
                                           country_code,
                                           type,
                                           lat,
                                           lon,
                                           population)
                        SELECT COALESCE(other_names['name:en'], name), -- prefer english name
                               -- prefix with english name, if available
                               CONCAT(COALESCE(other_names['name:en'], name), '; ', display_name),
                               '{country_code}',
                               type,
                               location[2],
                               location[1],
                               population 
                        FROM read_json('{tmp}',
                                       ignore_errors=true,
                                       format=newline_delimited,
                                       columns={{
                                          name:'VARCHAR',
                                          other_names:'MAP(VARCHAR, VARCHAR)',
                                          display_name:'VARCHAR',
                                          type:'VARCHAR',
                                          location:'DOUBLE[]',
                                          population:'INTEGER'}})
                        WHERE name IS NOT NULL
                        AND population IS NOT NULL 
                        """)
    seconds = time.time() - now
    place_count = count(db, "SELECT COUNT(*) FROM places")
    logger.info(f"Loaded {place_count} locations in {seconds} seconds")

def start(
    database: DatabaseFileType = Path("images.db"),
):
    load_zip_files(database)


def main():
    typer.run(start)

if __name__ == "__main__":
    typer.run(start)
