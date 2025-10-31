from takeout import load
import datetime
import pandas as pd
import duckdb


def test_date_from_filename():
    date_values = {
        "999920250203_010203xyzzy": datetime.datetime(2025, 2, 3, 1, 2, 3),
        "9999/20250203xyzzy": datetime.datetime(2025, 2, 3),
        "Pictures from 2025/0203xyzzy": datetime.datetime(2025, 1, 1),
        "Pictures from 2024-02-04/0203xyzzy": datetime.datetime(2024, 2, 4),
        "/Not a date 9876-01-01": None,
    }
    for k, v in date_values.items():
        dt = load.date_from_filename(k)
        assert dt == v


def test_words_of_filename():
    words = load.words_of_filename("/foo/bar/trip to Tutankhamun tomb/img.jpg")
    assert set(words) == {"tutankhamun", "tomb"}


def test_extract_meta():
    image = "tests/2025-02-04 some.jpg"
    testfile = image + ".supplemental-metadata.json"
    meta = load.process_metadata((image, testfile, ""))
    expected = {
        "path": "tests/2025-02-04 some.jpg",
        "taken": datetime.datetime(2014, 10, 19, 9, 42, 39),
        "lat": 1.0,
        "lon": 2.0,
        "words": ["tests"],
    }
    assert sorted(meta.items()) == sorted(expected.items())


def test_fetch_many():
    db = duckdb.connect(":memory:")
    db.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
    for i in range(1000):
        db.execute("insert into test(id, name) values(?, ?)", [i, str(i)])
    for mod in 2, 3:
        rows = load.fetch_many(
            db, "SELECT id, name FROM test WHERE id % ? = 0 ORDER by id asc", [mod]
        )
        for i, (id, name) in enumerate(rows):
            n = i * mod
            assert id == n
            assert name == str(n)
    assert [] == list(
        load.fetch_many(db, "SELECT id, name FROM test WHERE name = 'missing'")
    )


def test_is_small():
    def is_small(*coords):
        return load.is_small(coords)

    assert is_small(1, 1, 1, 1)
    assert is_small(1, 1, 1000, 10)
    assert not is_small(1, 1, 1000, 500)
