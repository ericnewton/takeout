from takeout.fix_logging import fix_logging
from takeout import db
from takeout import sql
import duckdb

fix_logging()

def test_fetchmany():
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
    for i in range(1000):
        conn.execute("INSERT INTO test(id, name) VALUES(?, ?)", [i, str(i)])
    for mod in 2, 3:
        rows = db.fetchmany(
            conn, "SELECT id, name FROM test WHERE id % ? = 0 ORDER BY id ASC", [mod]
        )
        for i, (id, name) in enumerate(rows):
            n = i * mod
            assert id == n
            assert name == str(n)
    assert [] == list(
        db.fetchmany(conn, "SELECT id, name FROM test WHERE name = 'missing'")
    )
    conn.close()

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
    conn.close()


def test_schema():
    conn = duckdb.connect(":memory:")
    sql.create_tables(conn)
    sql.create_location_table(conn)

    # try to run all the queries against an empty database
    #
    # This is a quick check agains the schema and the assigned bind
    # types.
    
    values = dict(lat=40.774194,
                  lon=-73.9697793,
                  prefix="paris",
                  limit=3,
                  hash="0"*64,
                  distance=10*1000.,
                  id=0,
                  location="Paris")
    for name in dir(sql):
        if name.isupper():
            query = getattr(sql, name)
            assert isinstance(query, sql.Query)
            # counts should return zero
            if name.endswith("_COUNT"):
                assert 0 == query.count(conn, -1)
            elif name == 'IMAGE_TOTAL_PROCESSED_COUNTS':
                assert (0, 0) == query.fetchone(conn)
            else:
                # just bind to some example data and run the query
                if not query.binds:
                    assert [] == query.fetchall(conn)
                else:
                    binds = []
                    for bname, _ in query.binds:
                        binds.append(values[bname])
                    assert [] == query.fetchall(conn, binds)
            
    conn.close()
