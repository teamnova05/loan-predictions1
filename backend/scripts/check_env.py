import os
from dotenv import load_dotenv

load_dotenv()  # loads backend/.env

print("ENV MONGO_URI:", os.getenv("MONGO_URI"))
print("ENV MONGO_DB_NAME:", os.getenv("MONGO_DB_NAME"))
