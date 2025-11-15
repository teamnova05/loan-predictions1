from config.db import get_db, client, MONGO_DB_NAME
import os

print("Using MONGO_URI:", os.getenv("MONGO_URI"))
print("Using DB name:", MONGO_DB_NAME)

db = get_db()
try:
    info = client.server_info()
    print("Mongo server info retrieved. version:", info.get("version"))
except Exception as e:
    print("ERROR: could not reach MongoDB server:", e)
    raise SystemExit(1)

try:
    # Access collection using dictionary syntax
    coll = db["__test_conn__"]
    res = coll.insert_one({"ping": "ok"})
    print("Inserted doc id:", res.inserted_id)
    docs = list(coll.find({}).limit(5))
    print("Docs in __test_conn__:", docs)
    # cleanup
    coll.delete_many({})
    print("Cleanup done.")
except Exception as e:
    print("ERROR during test insert/read:", e)
    raise SystemExit(1)

print("MongoDB connection & basic operations: SUCCESS")
