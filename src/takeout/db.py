import duckdb
from typing import Generator, Tuple, Iterable, Any

def count(db, query, default=0):
    row = db.execute(query).fetchone()
    if row:
        return row[0]
    return default

def fetch_many(
    db: duckdb.DuckDBPyConnection,
    query: str,
    bindings: Iterable[Any] = (),
    chunk: int = 100,
) -> Generator[Tuple, None, None]:
    with db.cursor() as cur:
        result = cur.execute(query, bindings)
        while True:
            rows = result.fetchmany(chunk)
            if not rows:
                break
            for row in rows:
                yield row


