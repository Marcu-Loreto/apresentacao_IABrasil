# db_funcs.py
import os
import json
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor

def get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD")
    )

def db_get_messages(session_id, limit=None):
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        if limit:
            cur.execute("SELECT role,content,timestamp,metadata FROM messages WHERE session_id=%s ORDER BY created_at DESC LIMIT %s", (session_id, limit))
        else:
            cur.execute("SELECT role,content,timestamp,metadata FROM messages WHERE session_id=%s ORDER BY created_at ASC", (session_id,))
        rows = cur.fetchall()
        msgs = [{"role": r["role"], "content": r["content"], "timestamp": r["timestamp"].isoformat(), "metadata": json.loads(r["metadata"]) if r["metadata"] else {}} for r in rows]
        if limit:
            msgs.reverse()
        cur.close()
        conn.close()
        return msgs
    except:
        return []