from .fix_logging import fix_logging
import duckdb
import os
import io
import json
import pycurl
import logging
import tempfile
import time
import typer
from pathlib import Path
from .types import DatabaseFileType
from .db import count, BatchInserter
from . import sql
from zipfile import ZipFile

fix_logging()

logger = logging.getLogger(__name__)
EMPTY = {}

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
    sql.create_location_table(db)

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

    COLS = [
        'name', 'display_name', 'country_code', 'type',
        'lat', 'lon', 'population'
    ]
    with tempfile.TemporaryDirectory() as tempdir, \
         BatchInserter(db, "places", COLS, 1000) as places:
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
                        with zf.open(name) as fp:
                            for line in fp:
                                obj = json.loads(line)
                                population = obj.get("population", None)
                                if population is None:
                                    continue
                                display_name = obj["display_name"]
                                type_ = obj["type"]
                                location = obj["location"]
                                lat = location[1]
                                lon = location[0]
                                
                                # name should be the first display name
                                first = display_name.split(", ", 1)[0]
                                
                                # but prefer the english name, if available
                                other_names = obj.get('other_names', EMPTY)
                                name = other_names.get("name:en", first)
                                
                                # jam the chosen name onto the display name
                                display_name = name + "; " + display_name,
                                
                                places.add_row(name,
                                               display_name,
                                               country_code,
                                               type_,
                                               lat,
                                               lon,
                                               population)

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
