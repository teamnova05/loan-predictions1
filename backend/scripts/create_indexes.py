# backend/scripts/create_indexes.py
from config.db import get_db

db = get_db()

print("Creating indexes on loan_requests...")
db["loan_requests"].create_index([("user_id", 1)])
db["loan_requests"].create_index([("created_at", -1)])
print("Indexes created.")
