# takeout - Index and view Google Photos after a Takeout request

--

How can you search and view your Google Images and video stored
locally?

Here's how:

* Perform a data takeout, selecting just Google Photos

** Browse to https://takeout.google.com/
** Select "Deselect All" so none of the Products are selected
** Select the Google Photos for export
** Select "Next step"
** Select file type (.zip, the default) and destination and size (see below)
** Select "Create export"

Wait.  This can take **days.**

* Download all the zip files.

If you choose "Send download link via email" you will need to manually
download all the zip files. I **highly** recommend you choose the
largest zip size (4G) to reduce the number of files you need to
download.

My last takeout of 45,000 images and videos required downloading 71 4G
zip files. This is not fun.

I pay for extra cloud storage in Google Drive. I direct the takeout
request to "Add to Drive" which will place the .zip files in Google
Drive storage. This is much easier to automate a download.

On Linux, using Gnome, I can download the **My Drive/Takeout**
directory using normal Gnome GUI file copying facilities.  That is, I
can navigate the Gnome file browsing to Google Drive, copy the Takeout
directory and paste it into a local file system. It's not necessarily
faster, but it is less tedious.

If you get errors downloading the many files, perhaps you can use
[rclone](https://rclone.org) to incrementally download files. It
includes detailed instructions for configuring rclone to talk to
Google Drive.

For example, I am able to fetch my latest takeout with this command
(after configuring `gdrive` to access my Google Drive):

```
$ rclone -v copy gdrive:Takeout .
```

* Pip install the software

```
$ uv pip install -e .
```

* Load the files into an "images.db" duckdb database

```
$ uv run load --takeout_directory /path/to/my/Takeout
```

Loading all the images/video will take some time.  May last load took
the better part of a day with an 8 core Ryzen 7 5700U mini pc.  Face
location and comparisons are CPU intensive.

Loading is incremental and fairly resilient to errors.  You can run it
over the zip files without unpacking them, or even waiting for all of
them to download. Just re-run the load command to pick up any new
files as the download makes progress.

* Fetch location data.

```
$ uv run load_locations
```

This command downloads and inserts the name, location and populations
for a half million cities, towns and villages worldwide.  It takes
about 5 minutes, including download time on my slow (40mbs) home
network.

* Browse the data:

```
$ uv run server
```

Point your web browser to http://localhost:8080/

## Why? Can't I just browse my photos online using Google Photos?

Sure, yeah.  But it's a little slow to browse. You can view the
thumbnails of the images very quickly when running locally.  The
entire duckdb database containing all the thumbnails for all 45,000
images in my Google Photos account is only 1.1G.  That fits in memory.

Google's image (and face) recognition really shines.  If you search
for "mountain", you're likely to find a picture with a mountain.
That's pretty magical and this software is **not** that good. But you
can search for "Denver", for example, and you'll get thumbnails for
all the images of those mountains you saw on your last visit.

Is it better to search for "Paris" and see a thousand thumbnails in a
second, or "Eiffel Tower" and see 30 larger, but very specific, images
dribble in?  Try it and see.


## Goal: including all the images I have that pre-date Google Photos

I'd like to pull in all the photos I have that aren't presently
uploaded to Photos.  I need to add:

* Tools to extract metadata from various image formats

### TODO

* Display image metadata, location information with the image
* Support quick next/prev image from query
* Support "next page" at the end of a query
