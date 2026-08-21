import hashlib
import sqlite3
from pathlib import Path


class TranslationCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS translations (
                cache_key TEXT PRIMARY KEY,
                source_text TEXT NOT NULL,
                source_language TEXT NOT NULL,
                target_language TEXT NOT NULL,
                model TEXT NOT NULL,
                translation TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    @staticmethod
    def key(text, source_language, target_language, model):
        raw = f"v9-tablecells|{source_language}|{target_language}|{model}|{text}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def get(self, text, source_language, target_language, model):
        key = self.key(text, source_language, target_language, model)
        row = self.conn.execute(
            "SELECT translation FROM translations WHERE cache_key=?",
            (key,),
        ).fetchone()
        return row[0] if row else None

    def put(self, text, source_language, target_language, model, translation):
        key = self.key(text, source_language, target_language, model)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO translations
            (cache_key, source_text, source_language, target_language, model, translation)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (key, text, source_language, target_language, model, translation),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
