from takeout.fix_logging import fix_logging
from takeout import db
import duckdb

fix_logging()

def test_fetch_many():
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
    for i in range(1000):
        conn.execute("INSERT INTO test(id, name) VALUES(?, ?)", [i, str(i)])
    for mod in 2, 3:
        rows = db.fetch_many(
            conn, "SELECT id, name FROM test WHERE id % ? = 0 ORDER BY id ASC", [mod]
        )
        for i, (id, name) in enumerate(rows):
            n = i * mod
            assert id == n
            assert name == str(n)
    assert [] == list(
        db.fetch_many(conn, "SELECT id, name FROM test WHERE name = 'missing'")
    )

def test_batch_insert():
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE test2 (id INTEGER PRIMARY KEY, name VARCHAR)")

    COUNT_QUERY = "SELECT COUNT(*) FROM test2"
    with conn.cursor() as cur:
        bi = db.BatchInserter(cur, 'test2', ['id', 'name'], 2)
        bi.add_row(1, "one")
        bi.add_row(2, "two")
        assert 0 == db.count(cur, COUNT_QUERY)
        bi.add_row(3, "three")
        assert 3 == db.count(cur, COUNT_QUERY)
        bi.add_row(4, "four")
        assert 3 == db.count(cur, COUNT_QUERY)
        bi.flush()
        assert 4 == db.count(cur, COUNT_QUERY)

