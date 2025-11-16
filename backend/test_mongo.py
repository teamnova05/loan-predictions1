from pymongo import MongoClient
import certifi

uri = "mongodb+srv://teamnova464_db_user:novateam%4005@cluster0.peaw8aj.mongodb.net/smartloandb?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(uri, tlsCAFile=certifi.where())

try:
    client.admin.command("ping")
    print("MongoDB Connected Successfully!")
except Exception as e:
    print("Connection failed:", e)
