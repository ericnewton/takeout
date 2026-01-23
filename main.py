#!/usr/bin/env python3

from takeout.types import DatabaseFileType
from takeout import sql

database: DatabaseFileType = DatabaseFileType("images.db")
from nicegui import app, ui
import duckdb

db: duckdb.DuckDBPyConnection
YEARS: list[str]

async def init_db() -> None:
    print('starting')
    global db
    global YEARS
    db = duckdb.connect(database, read_only=True)
    all_year_strings = [str(year) for year, in sql.YEARS.fetchall(db)]
    YEARS = all_year_strings[1:-1]


async def close_db() -> None:
    print('stopping')

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

@ui.page('/')
async def index():
    async def search() -> None:
        print(f'Place {place.value} distance {distance.value} words {words.value}')

    async def update_selection(e) -> None:
        print(e.value)
        if e.value and len(e.value) >= 3:
            options = location_complete(e.value)
            place.set_options(options)
            if options:
                place.set_value(options[0])

    with ui.row():
        filter = ui.input(label='Filter Place Name', on_change=update_selection)
        place = ui.select(options=["Place"], label='Place').classes('w-100')
    distance = ui.number(label='Distance', format='%.0f', value=10)
    words = ui.input(label='Key Words')
    ui.button(on_click=search, icon='search')

ui.run()
