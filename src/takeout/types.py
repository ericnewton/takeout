from typing_extensions import Annotated
import typer
from pathlib import Path

TakeoutDirectoryType = Annotated[
    Path,
    typer.Option(
        exists=True, dir_okay=True, file_okay=False, readable=True
    ),
]
DatabaseFileType = Annotated[Path, typer.Option(dir_okay=False, file_okay=True)]
