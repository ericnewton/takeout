from .fix_logging import fix_logging
from .types import DatabaseFileType
from .io import InputFile
from . import sql
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse, Response
import duckdb
import magic
import logging
import typer
from typing import Union, Any, Annotated, Optional
from pathlib import Path

fix_logging()


app = FastAPI()
templates = Jinja2Templates(directory="templates")
db: duckdb.DuckDBPyConnection
YEARS: list[str]

logger = logging.getLogger(__name__)


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


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("search.html",
                                      dict(request=request,
                                           data={},
                                           results=SearchResult(None),
                                           faces=common_faces(),
                                           years=YEARS))


@app.get("/location_complete")
def location_complete(term: str) -> Any:
    MAX_LOCATIONS = 40
    if not term:
        return []
    rows = None
    with db.cursor() as cur:
        try:
            rows = sql.LOCATION_COMPLETION_QUERY.fetchall(cur, [term, MAX_LOCATIONS])
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


def common_faces(selected_faces: set[int] = set()) -> dict[int, bool]:
    "Fetch the faces that matched the most images"
    with db.cursor() as cur:
        rows = sql.FACE_QUERY.fetchall(cur)
        if rows:
            return {id: id in selected_faces for id, count in rows}
        return {}


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request,
           distance: Annotated[Optional[float], Form()] = None,
           location: Annotated[str, Form()] = '',
           faces: Annotated[list[int], Form()] = [],
           words: Annotated[str, Form()] = '',
           before: Annotated[Optional[int], Form()] = None,
           ):
    location = location.strip()
    selected_faces = set(faces)
    results = SearchResult(None)
    ands = ""
    binds = []
    cur = db.cursor()
    if distance not in [None, ""] and distance > 0:
        rows = sql.LOCATION_LOOKUP_QUERY.fetchone(cur, [location, location])
        if rows:
            name, lat, lon = rows
            distance_m = 1000. * distance  # convert km to m
            ands += """
               AND i.lat is not null
               AND ST_Distance_Spheroid(ST_Point(i.lat, i.lon),
                                        ST_Point(?, ?)) < ?
            """
            binds += [lat, lon, distance_m]
    if selected_faces:
        ands += """
               AND i.path IN (
                  SELECT f.path FROM face_matches f WHERE f.face_id IN ?
               )
        """
        binds += [list(selected_faces)]
    if before:
        year = before
        ands += f"""
               AND i.taken < date '{year}-01-01'
        """
    if words:
        word_list = [w for w in words.strip().split(" ") if len(w) > 1]
        ands += """
           AND LIST_HAS_ANY(i.words, ?)
        """
        binds += [word_list]

    # dedup search results by hash
    query = f"""
    INSTALL spatial;
    LOAD    spatial;        
    SELECT i.hash,
           ARBITRARY(i.taken) t
      FROM images i
     WHERE i.hash IS NOT null {ands}
     GROUP BY i.hash
     ORDER BY t DESC
     LIMIT ?
            """
    binds += [SearchResult.MAX + 1]
    logger.info(f"query = {query}, binds = {binds}")
    rows = cur.execute(query, binds)
    results = SearchResult(rows)
    return templates.TemplateResponse(
        "search.html",
        dict(
            request=request,
            data=dict(distance=distance,
                      location=location,
                      faces=faces,
                      words=words,
                      before=str(before)),
            results=results,
            faces=common_faces(selected_faces),
            years=YEARS,
        ))


@app.get("/thumb/{hash}")
def thumb(hash: str):
    with db.cursor() as cur:
        row = sql.THUMBNAIL_QUERY.fetchone(cur, [hash])
        if row is None:
            return ""
        (img,) = row
        return Response(content=img, headers={"Content-Type": "image/jpeg"})


def response(data: bytes) -> Response:
    # TODO: cache control
    return Response(content=data,
                    headers={"Content-Type": magic.from_buffer(data, mime=True)})


@app.get("/viewer/{hash}", response_class=HTMLResponse)
def viewer(request: Request, hash: str):
    with db.cursor() as cur:
        row = sql.IMAGE_DETAIL.fetchone(cur, [hash])
        if row is None:
            return ""
        path, archive, mimetype, size, width, height, taken, words, lat, lon = row
        data = dict(hash=hash,
                    path=path,
                    mimetype=mimetype,
                    archive=archive,
                    size=size,
                    width=width,
                    height=height,
                    taken=taken,
                    lat=lat,
                    lon=lon,
                    words=words)
        if lat and lon:
            row = sql.NEAREST_LOCATION.fetchone(cur, [lat, lon, 10 * 1000.])
            if row:
                data["location"] = row[0]
        rows = sql.FACES_FOR_IMAGE.fetchall(cur, [hash])
        if rows:
            data["faces"] = [face_id for face_id, in rows]
    return templates.TemplateResponse("viewer.html",
                                      dict(request=request,
                                           data=data))

@app.get("/image/{hash}", response_class=Response)
def image(hash: str):
    with db.cursor() as cur:
        row = sql.IMAGE_INPUT.fetchone(cur, [hash])
        if row is None:
            return ""
        (path, archive) = row
    t = InputFile(path, archive)
    with t.open() as fp:
        return response(fp.read())


@app.get("/face/{id}", response_class=Response)
def face(id: str):
    with db.cursor() as cur:
        row = sql.FETCH_FACE.fetchone(cur, [int(id)])
        if row is None:
            return ""
        (image,) = row
    return response(image)


@app.get("/favicon.ico", include_in_schema=False, response_class=FileResponse)
def favicon():
    return FileResponse("static/favicon.png")


def run(
    port: int = 8080,
    bind: str = "127.0.0.1",
    database: DatabaseFileType = Path("images.db"),
) -> None:
    global YEARS
    global db

    format = "%(asctime)s %(levelname)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO)

    db = duckdb.connect(database, read_only=True)

    all_year_strings = [str(year) for year, in sql.YEARS.fetchall(db)]
    YEARS = all_year_strings[1:-1]

    import uvicorn
    uvicorn.run(app, host=bind, port=port)


def main():
    typer.run(run)


if __name__ == "__main__":
    main()

