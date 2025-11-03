import duckdb
import pandas as pd
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

class BatchInserter:
    def __init__(self, db: duckdb.DuckDBPyConnection, table: str, columns: list[str], max: int = 100):
        self.db = db
        self.table = table
        self.columns = columns
        self.data: dict[str, list[Any]] = {c: [] for c in self.columns}
        self.count = 0
        self.max = max

    def flush(self) -> None:
        if self.data:
            col_str = ", ".join(self.columns)
            df = pd.DataFrame(self.data)  # noqa: F841
            stmt = f"""
                 INSERT OR IGNORE
                   INTO {self.table} ({col_str})
                 SELECT * from df
            """
            self.db.execute(stmt)
            self.data = {c: [] for c in self.columns}
            self.count = 0

    def add_row(self, *cols) -> None:
        for name, value in zip(self.columns, cols):
            self.data[name].append(value)
        self.count += 1
        if self.count > self.max:
            self.flush()


