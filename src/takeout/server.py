from .fix_logging import fix_logging
from .types import DatabaseFileType
from .io import TakeoutFile
from .db import fetch_many
import flask
import duckdb
import magic
import logging
import os
import typer
from typing import Union, Any
from pathlib import Path

fix_logging()


app = flask.Flask(__name__, template_folder=os.path.join(os.getcwd(), "templates"))
db: duckdb.DuckDBPyConnection
YEARS: list[str]

logger = logging.getLogger("server")


GET = ["GET"]
POST = ["POST"]


class SearchResult:
    "Organize search results by year, track if there are more than MAX results"

    MAX = 1000
    CHUNK = 100

    def __init__(self, cursor: Union[None, duckdb.DuckDBPyConnection]):
        self.years = []
        self.more = False
        current_year = None
        current_list = []
        count = 0
        if cursor:
            while True:
                rows = cursor.fetchmany(self.CHUNK)
                if not rows:
                    break
                for hash, taken in rows:
                    year = "Unknown"
                    if taken:
                        year = taken.year
                    if year != current_year:
                        current_list = []
                        current_year = year
                        self.years.append((year, current_list))
                    current_list.append(hash)
                    count += 1
                    if count > self.MAX:
                        self.more = True
                        break
        self.count = count


@app.route("/", methods=GET)
def index() -> str:
    return flask.render_template(
        "search.html", data={}, results=SearchResult(None), faces=faces(), years=YEARS
    )


@app.route("/location_complete", methods=GET)
def location_complete() -> Any:
    MAX_LOCATIONS = 40
    term = flask.request.args.get("term", "")
    if not term:
        return []
    rows = None
    with db.cursor() as cur:
        try:
            rows = cur.execute(
                """
              SELECT display_name
                FROM places
               WHERE display_name ILIKE CONCAT(?, '%')
               ORDER BY population DESC
               LIMIT ?
                """,
                [term, MAX_LOCATIONS],
            ).fetchall()
        except duckdb.CatalogException as e:
            logger.error(f"Location read error: data probably not loaded {e}")
        except Exception:
            logger.exception("Error reading location data")
    if rows is None:
        return [
            "You must load the location data! See README.md",
            "Try `$ uv run load_locations`",
        ]
    if not rows:
        rows = [("Nothing Found",)]
    return [r for (r,) in rows]


def faces(selected_faces: set[int] = set()):
    with db.cursor() as cur:
        rows = cur.execute(
            """
        SELECT face_id, c
          FROM (
            SELECT face_id, count(*) c
              FROM face_matches
             GROUP BY face_id
          )
         ORDER BY c DESC
         LIMIT 40
            """
        ).fetchall()
        faces = {}
        if rows:
            faces = {id: id in selected_faces for id, count in rows}
        return faces


@app.route("/search", methods=POST)
def search() -> str:
    data = flask.request.form
    distance = int(data["distance"])
    location = data["location"].strip()
    selected_faces = set(map(int, data.getlist("faces")))
    results = SearchResult(None)
    ands = ""
    binds = []
    cur = db.cursor()
    if distance > 0 and location != "":
        rows = cur.execute(
            """
            SELECT name, lat, lon
              FROM places
             WHERE display_name = ? OR name = ?
             ORDER BY population DESC LIMIT 1
            """,
            [location, location],
        )
        location_data = rows.fetchone()
        if location_data:
            name, lat, lon = location_data
            distance *= 1000  # distance km to m
            ands += """
               AND i.lat is not null
               AND ST_Distance_Spheroid(ST_Point(i.lat, i.lon),
                                        ST_Point(?, ?)) < ?
            """
            binds += [lat, lon, distance]
    if selected_faces:
        ands += """
               AND i.path IN (
                  SELECT f.path FROM face_matches f WHERE f.face_id IN ?
               )
        """
        binds += [list(selected_faces)]
    if data["before"]:
        year = int(data["before"])
        ands += f"""
               AND i.taken < date '{year}-01-01'
        """
    if data["words"]:
        words = [w for w in data["words"].strip().split(" ") if len(w) > 1]
        ands += """
           AND list_has_any(i.words, ?)
        """
        binds += [words]

    query = f"""
    INSTALL spatial;
    LOAD    spatial;        
    SELECT i.hash,
           i.taken
      FROM images i
     WHERE i.hash IS NOT null {ands}
     ORDER BY taken DESC
     LIMIT ?
            """
    binds += [SearchResult.MAX + 1]
    logger.info(f"query = {query}, binds = {binds}")
    rows = cur.execute(query, binds)
    results = SearchResult(rows)
    return flask.render_template(
        "search.html",
        data=data,
        results=results,
        faces=faces(selected_faces),
        years=YEARS,
    )


@app.route("/thumb/<hash>", methods=GET)
def thumb(hash: str) -> Union[flask.Response, str]:
    with db.cursor() as cur:
        row = cur.execute(
            """
               SELECT thumbnail
                 FROM images
                WHERE hash = ?
            """,
            [hash],
        ).fetchone()
        if row is None:
            return ""
        (img,) = row
        response = flask.make_response(img)
        response.headers.set("Content-Type", "image/jpeg")
        return response


def response(data: bytes) -> flask.Response:
    result = flask.make_response(data)
    result.headers.set("Content-Type", magic.from_buffer(data, mime=True))
    result.cache_control.max_age = 60 * 60 * 24
    return result


@app.route("/image/<hash>", methods=GET)
def image(hash: str) -> Union[flask.Response, str]:
    with db.cursor() as cur:
        row = cur.execute(
            """SELECT path, archive FROM images WHERE hash = ?""", [hash]
        ).fetchone()
        if row is None:
            return ""
        (path, archive) = row
    t = TakeoutFile(path, archive)
    with t.open() as fp:
        return response(fp.read())


@app.route("/face/<id>", methods=GET)
def face(id: str) -> Union[flask.Response, str]:
    with db.cursor() as cur:
        row = cur.execute("SELECT image FROM faces WHERE id = ?", [int(id)]).fetchone()
        if row is None:
            return ""
        (image,) = row
    return response(image)


@app.route("/favicon.ico")
def favicon() -> flask.Response:
    return flask.send_from_directory("static", "favicon.png", mimetype="image/png")


def run(
    port: int = 8080,
    debug: bool = False,
    bind: str = "127.0.0.1",
    database: DatabaseFileType = Path("images.db"),
) -> None:
    global YEARS
    global db

    format = "%(asctime)s %(levelname)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO)

    db = duckdb.connect(database, read_only=True)
        
    q = "SELECT year(taken) y FROM images GROUP BY y ORDER BY y desc"
    YEARS = [str(year) for year, in fetch_many(db, q)][1:-1]

    app.run(bind, port, debug=debug)


def main():
    typer.run(run)


if __name__ == "__main__":
    typer.run(run)
