import os
import duckdb
from pathlib import Path
from typing import Optional


class Config:
    def __init__(
        self,
        takeout_directory: Path = Path("."),
        database: Path = Path("images.db"),
        concurrency: int = 0,
        scan_faces: bool = True
    ):
        self.db: Optional[duckdb.DuckDBPyConnection] = None
        self.takeout_directory = takeout_directory
        self.database = database
        self.scan_faces = scan_faces
        if concurrency:
            self.concurrency = concurrency
        else:
            cpus = os.cpu_count()
            self.concurrency = int(cpus * 1.5) if cpus else 4

    def cursor(self) -> duckdb.DuckDBPyConnection:
        if not self.db:
            self.db = duckdb.connect(self.database)
        return self.db.cursor()
