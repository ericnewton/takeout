import logging
import warnings
from pathlib import Path
from typing import Optional

# silence warnings
logging.captureWarnings(True)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="face_recognition_models"
)
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, module="moviepy")

logging_fixed: bool = False

def fix_logging(log: Optional[Path] = None):
    "a no-op so we don't get warnings about an unused import when importing this module"
    global logging_fixed
    if logging_fixed:
        return
    log_format = "%(asctime)s %(levelname)s: %(message)s"
    if log:
        logging.basicConfig(format=log_format, level=logging.INFO, filename=log, filemode="a")
    else:
        logging.basicConfig(format=log_format, level=logging.INFO)
    logging_fixed = True

