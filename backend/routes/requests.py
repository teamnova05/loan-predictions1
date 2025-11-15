from flask import Blueprint, request, jsonify
from config.db import get_db

req_bp = Blueprint("requests", __name__)
db = get_db()

@req_bp.route("/loan-requests", methods=["GET"])
def list_requests():
    # optional filter by user_id
    user_id = request.args.get("user_id")
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 25))
    skip = (page - 1) * per_page

    query = {}
    if user_id:
        query["user_id"] = user_id

    docs = db["loan_requests"].find(query).sort("created_at", -1).skip(skip).limit(per_page)
    items = []
    for d in docs:
        d["_id"] = str(d["_id"])
        items.append(d)
    return jsonify(items), 200
