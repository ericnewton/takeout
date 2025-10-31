import logging
import warnings

# importing this module does all the work

log_format = "%(asctime)s %(levelname)-8s: %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)
logging.captureWarnings(True)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="face_recognition_models"
)
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, module="moviepy")

def fix_logging():
    "a no-op so we don't get warnings about an unused import when importing this module"
    pass
