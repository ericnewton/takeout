import os
from pathlib import Path


class Config:
    def __init__(
        self,
        takeout_directory: Path = Path("."),
        database: Path = Path("images.db"),
        concurrency: int = 0,
        scan_faces: bool = True,
        log: Path = Path("takeout.log")
    ):
        self.takeout_directory = takeout_directory
        self.database = database
        self.scan_faces = scan_faces
        self.log = log
        if concurrency:
            self.concurrency = concurrency
        else:
            cpus = os.cpu_count()
            self.concurrency = int(cpus * 1.5) if cpus else 4
