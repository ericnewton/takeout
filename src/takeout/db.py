import duckdb
import pandas as pd
from typing import Generator, Tuple, Iterable, Any
import logging

type database: duckdb.DuckDBPyConnection

logger = logging.getLogger("db")

def count(db, query, default=0):
    """
    Perform a count query, returning the count, or the default if
    the query returns nothing
    """
    rows = db.execute(query).fetchall()
    if rows:
        return rows[0][0]
    return default

def fetch_many(
    db: duckdb.DuckDBPyConnection,
    query: str,
    bindings: Iterable[Any] = (),
    chunk: int = 100,
) -> Generator[Tuple, None, None]:
    "Return query results using a generator"
    with db.cursor() as cur:
        result = cur.execute(query, bindings)
        while True:
            rows = result.fetchmany(chunk)
            if not rows:
                break
            for row in rows:
                yield row

class BatchInserter:
    "Insert rows into a table in batches for efficiency"
    
    def __init__(self, db: duckdb.DuckDBPyConnection, table: str, columns: list[str], max: int = 100):
        self.db = db
        self.table = table
        self.columns = columns
        self.data: dict[str, list[Any]] = {c: [] for c in self.columns}
        self.count = 0
        self.max = max

    def __enter__(self) -> "BatchInserter":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def close(self) -> None:
        self.flush()

    def flush(self) -> None:
        if self.count > 0:
            col_str = ", ".join(self.columns)
            df = pd.DataFrame.from_dict(self.data)  # noqa: F841
            stmt = f"""
                 INSERT OR IGNORE
                   INTO {self.table} ({col_str})
                 SELECT * from df
            """
            self.db.execute(stmt)
            self.data = {c: [] for c in self.columns}
            self.count = 0

    def row_added(self):
        self.count += 1
        if self.count > self.max:
            self.flush()
        
    def add_row(self, *cols) -> None:
        for name, value in zip(self.columns, cols):
            self.data[name].append(value)
        self.row_added()

    def add_dict(self, d: dict[str, Any]) -> None:
        for name in self.columns:
            self.data[name].append(d.get(name, None))
        self.row_added()
