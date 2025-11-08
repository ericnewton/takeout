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

LOG_FORMAT = "%(asctime)s %(levelname)s: %(message)s"

def fix_logging(log: Optional[Path] = None):
    "a no-op so we don't get warnings about an unused import when importing this module"
    global logging_fixed
    if logging_fixed:
        return
    if log:
        logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, filename=log, filemode="a")
    else:
        logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
    logging_fixed = True

