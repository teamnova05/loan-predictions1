import os
from dotenv import load_dotenv

# load .env from backend folder
load_dotenv()

from pymongo import MongoClient

# Read env vars (after load_dotenv)
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "smartloandb")

print("Loaded MONGO_URI (truncated):", (MONGO_URI[:120] + '...') if MONGO_URI else None)
print("Loaded MONGO_DB_NAME:", MONGO_DB_NAME)

client = None
try:
    # If MONGO_URI is None, this connects to localhost
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()  # force a connection test
    print("MongoDB: connected successfully!")
except Exception as e:
    print("MongoDB connection FAILED:", e)

def get_db(name=None):
    if client is None:
        raise RuntimeError("MongoDB client is not initialized. Check MONGO_URI and network.")
    return client[name or MONGO_DB_NAME]
