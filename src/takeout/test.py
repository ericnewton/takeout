import zipfile
from PIL import Image

with zipfile.ZipFile(
    "/media/ecn/homes/ecn/takeouts/takeout-20251019/takeout-20251019T215950Z-1-001.zip"
) as zf:
    with zf.open("Takeout/Google Photos/Trip to BZ/DSCF3259.JPG") as fp:
        with Image.open(fp) as im:
            width, height = im.size
            print((width, height))
