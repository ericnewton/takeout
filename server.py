#!/usr/bin/env python3

from takeout.types import DatabaseFileType, ListOrSet
from takeout.load import ITHUMBNAIL
from takeout import sql
from takeout.io import InputFile

from nicegui import app, ui
from fastapi.responses import Response
import magic
import duckdb
import logging
import typer
from typing import Any, Optional
import math
import argparse
from pathlib import Path

TITLE="Takeout"
THUMBNAIL_WIDTH = ITHUMBNAIL[0]
logger = logging.getLogger(__name__)
db: duckdb.DuckDBPyConnection
YEARS: list[str]
database_path: DatabaseFileType = Path("images.db")
PREVIEW_SIZE=600

class SearchResult:
    "Organize search results by year, track if there are more than MAX results"

    MAX = 500
    CHUNK = 100

    def __init__(self, cursor: duckdb.DuckDBPyConnection):
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

class ArgumentParser(argparse.ArgumentParser):
    def exit(self):
        raise ui.Exit
        
async def init_db() -> None:
    parser = ArgumentParser(description=TITLE)
    parser.add_argument('--debug', action='store_true', help='enable debugging log messages')
    parser.add_argument('--database', help='path to the image database', default=database_path)
    args = parser.parse_args()
    
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=level)
    
    global db
    db = duckdb.connect(args.database, read_only=True)
    
    global YEARS
    all_year_strings = [str(year) for year, in sql.YEARS.fetchall(db)]
    YEARS = [''] + all_year_strings[1:-1]

async def close_db() -> None:
    db.close()

app.on_startup(init_db)
app.on_shutdown(close_db)

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

def response(data: bytes) -> Response:
    # TODO: cache control
    return Response(content=data,
                    headers={"Content-Type": magic.from_buffer(data, mime=True)})

@app.get("/face/{id}", response_class=Response)
def face(id: str):
    "return an image for a face"
    with db.cursor() as cur:
        row = sql.FETCH_FACE.fetchone(cur, [int(id)])
        if row is None:
            return ""
        (image,) = row
    return response(image)

@app.get("/thumb/{hash}")
def thumb(hash: str):
    "return the thumbnail for an image"
    with db.cursor() as cur:
        row = sql.THUMBNAIL_QUERY.fetchone(cur, [hash])
        if row is None:
            return ""
        (img,) = row
        return response(img)

@app.get("/image/{hash}")
def image(hash: str):
    "return an image"
    with db.cursor() as cur:
        row = sql.IMAGE_INPUT.fetchone(cur, [hash])
        if row is None:
            return ""
        (path, archive) = row
    t = InputFile(path, archive)
    with t.open() as fp:
        return response(fp.read())

def common_faces() -> set[int]:
    "Fetch the the most common face ids"
    with db.cursor() as cur:
        rows = sql.FACE_QUERY.fetchall(cur)
        if rows:
            return {id for id, count in rows}
        return set()

def find_images(location: str,
                distance: Optional[int],
                words: list[str],
                before: Optional[int],
                faces: ListOrSet[int]) -> SearchResult:
    ands = ""
    binds = []
    cur = db.cursor()
    if distance and location not in [None, ""]:
        rows = sql.LOCATION_LOOKUP_QUERY.fetchone(cur, [location, location])
        if rows:
            name, lat, lon = rows
            meters = 1000. * distance  # convert km to m
            ands += """
               AND i.lat is not null
               AND ST_Distance_Spheroid(ST_Point(i.lat, i.lon),
                                        ST_Point(?, ?)) < ?
            """
            binds += [lat, lon, meters]
    if faces:
        ands += """
               AND i.path IN (
                  SELECT f.path FROM face_matches f WHERE f.face_id IN ?
               )
        """
        binds += [[f for f in faces]]
    if before:
        year = before
        ands += f"""
               AND i.taken < date '{year}-01-01'
        """
    if words:
        ands += """
           AND LIST_HAS_ANY(i.words, ?)
        """
        binds += [words]

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
    return SearchResult(rows)
    

@ui.page("/viewer/{hash}")
async def viewer(hash: str):

    await ui.context.client.connected()
    
    async def handle_key(e):
        all_image_hashes = app.storage.tab.get('images', [])
        hash = app.storage.tab.get('current_image', None)
        if not hash:
            return
        if e.action.keydown:
            try:
                offset = all_image_hashes.index(hash)
            except ValueError:
                return
            if e.key.arrow_left:
                offset -= 1
            if e.key.arrow_right:
                offset += 1
            offset %= len(all_image_hashes)
            await view(all_image_hashes[offset])

    async def view(hash):
        app.storage.tab['current_image'] = hash
        container.clear()
        with container:
            with db.cursor() as cur:
                row = sql.IMAGE_DETAIL.fetchone(cur, [hash])
                if row is None:
                    return ""
                path, archive, mimetype, size, width, height, taken, words, lat, lon = row
                faces = location = None
                if lat and lon:
                    row = sql.NEAREST_LOCATION.fetchone(cur, [lat, lon, 10 * 1000.])
                    if row:
                        location = row[0]
                rows = sql.FACES_FOR_IMAGE.fetchall(cur, [hash])
                if rows:
                    faces = [face_id for face_id, in rows]
                with ui.row():
                    if mimetype.split("/")[0] == "video":
                        ui.video(f"/image/{hash}").style(f"width: {PREVIEW_SIZE}px; height: {PREVIEW_SIZE}px")
                    else:
                        with ui.link(target=f"/image/{hash}"):
                            ui.image(f"/image/{hash}").style(f"width: {PREVIEW_SIZE}px")
                    with ui.grid(columns='auto auto'):
                        ui.label("Path")
                        ui.label(path)

                        if archive:
                            ui.label("Archive")
                            ui.label(archive)

                        ui.label("Size")
                        ui.label(f"{size} bytes")

                        ui.label("MIME type")
                        ui.label(mimetype).style('font-family: fixed')

                        ui.label("Taken")
                        ui.label(taken)

                        ui.label(f"Width x Height")
                        ui.label(f"{width} x {height}")

                        if lat:
                            ui.label("Location")
                            with ui.link(target=f'https://maps.google.com/?q={lat},{lon}'):
                                ui.label(f"{lat}, {lon}")

                        if words:
                            ui.label("Keywords:")
                            ui.label(", ".join(words))

                        if location:
                            ui.label("Near:")
                            ui.label(location)

                        if faces:
                            ui.label("Faces:")
                            with ui.row():
                                for face in faces:
                                    ui.image(f"/face/{face}").style(f"width: 64px")

    
    keyboard = ui.keyboard(on_key=handle_key)
    container = ui.row()
    await view(hash)
    

@ui.page('/')
async def index():
    await ui.context.client.connected()
    query = app.storage.tab.get('query', {})
    
    async def search() -> None:
        faces : set[int] = set({})
        for id in selected_faces:
            if selected_faces[id].value:
                faces.add(id)
        word_list = [w for w in words.value.strip().split(" ") if len(w) > 1]
        km = None
        if distance.value:
            km = int(distance.value)
        before = None
        if year.value:
            before = int(year.value)
        result = find_images(place.value, km, word_list, before, faces)
        query.update(dict(place=place.value, distance=km, words=word_list, year=before, faces=faces))
        await ui.context.client.connected()
        app.storage.tab['query'] = query
        images.clear()
        all_image_hashes = []
        with images:
            if not result.years:
                ui.label("There were no matching images")
            for year_label, hashes in result.years:
                ui.label(f'{year_label}').classes("font-bold text-2xl")
                ui.separator()
                with ui.row():
                    for hash in hashes:
                        with ui.link(target=f'/viewer/{hash}'):
                            ui.image(f"/thumb/{hash}").style(f"width: {THUMBNAIL_WIDTH}px")
                            all_image_hashes.append(hash)
            if result.more:
                ui.label("There are more results")
        app.storage.tab['images'] = all_image_hashes
        

    async def update_locations(e) -> None:
        "update locations as the user types in the location filter, default to the first element"
        if e.value and len(e.value) >= 3:
            options = location_complete(e.value)
            place.set_options(options)
            if options:
                place.set_value(options[0])
                query['place_options'] = options
                query['place_filter'] = e.value
                query['place'] = options[0]
        if not e.value:
            place.set_options([""])
            query['place_options'] = [""]
            query['place_filter'] = ""
            query['place'] = ""


    async def update_faces_dropdown(e):
        "Update the count in the selected faces dropdown button"
        count  = len([id for id in selected_faces if selected_faces[id].value])
        faces_dropdown.text = f'Faces ({count})'
        

    with ui.row():
        filter = ui.input(label='Filter Place Name', on_change=update_locations, value=query.get('place_filter', None))
        place = ui.select(options=query.get('place_options', ["Place"]), label='Place', value=query.get('place', None)).classes('w-100')
    distance = ui.number(label='Distance (km)', format='%.0f', value=query.get('distance', 10))
    words = ui.input(label='Key Words', value=' '.join(query.get('words', [])))
    query_year = query.get('year', None)
    if query_year:
        query_year = str(query_year)
    year = ui.select(options=YEARS, label='Before', value=query_year).classes('w-25')
    query_faces = query.get('faces', set())
    with ui.dropdown_button(f'Faces ({len(query_faces)})') as faces_dropdown:
        selected_faces = {}
        all_faces = list(common_faces())
        columns = len(all_faces)
        if columns > 10:
            columns = int(math.sqrt(columns))
        with ui.grid(columns=columns).classes('gap-0') as faces:
            for id in all_faces:
                with ui.checkbox('', on_change=update_faces_dropdown, value=id in query_faces) as cb:
                    selected_faces[id] = cb
                    image = ui.image(f'/face/{id}').classes('w-16')
    ui.button(on_click=search, icon='search')
    images = ui.column()

ui.run(title=TITLE, favicon="📷")
