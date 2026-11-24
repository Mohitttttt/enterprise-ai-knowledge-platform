import sqlite3
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATABASE_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "conversations.db"
)


class ConversationService:

    def __init__(self, database_path: Path = DATABASE_PATH):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.initialize_database()

    def get_connection(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize_database(self):
        with self.get_connection() as connection:

            connection.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                )
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(id)
                        ON DELETE CASCADE
                )
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS
                idx_messages_conversation
                ON messages(conversation_id, id)
            """)

    def create_conversation(self) -> dict[str, Any]:

        conversation_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        with self.get_connection() as connection:
            connection.execute(
                """
                INSERT INTO conversations (id, created_at)
                VALUES (?, ?)
                """,
                (conversation_id, created_at),
            )

        return {
            "id": conversation_id,
            "created_at": created_at,
        }

    def conversation_exists(self, conversation_id: str) -> bool:

        with self.get_connection() as connection:
            result = connection.execute(
                """
                SELECT id
                FROM conversations
                WHERE id = ?
                """,
                (conversation_id,),
            ).fetchone()

        return result is not None

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> dict[str, Any]:

        if role not in {"user", "assistant"}:
            raise ValueError(
                "Role must be either 'user' or 'assistant'."
            )

        if not content or not content.strip():
            raise ValueError("Message content cannot be empty.")

        created_at = datetime.now(timezone.utc).isoformat()

        with self.get_connection() as connection:

            cursor = connection.execute(
                """
                INSERT INTO messages (
                    conversation_id,
                    role,
                    content,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    role,
                    content.strip(),
                    created_at,
                ),
            )

            message_id = cursor.lastrowid

        return {
            "id": message_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content.strip(),
            "created_at": created_at,
        }

    def get_messages(
        self,
        conversation_id: str,
    ) -> list[dict[str, Any]]:

        with self.get_connection() as connection:

            rows = connection.execute(
                """
                SELECT id, conversation_id, role, content, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (conversation_id,),
            ).fetchall()

        return [dict(row) for row in rows]

    def delete_conversation(self, conversation_id: str) -> bool:

        with self.get_connection() as connection:

            connection.execute("PRAGMA foreign_keys = ON")

            cursor = connection.execute(
                """
                DELETE FROM conversations
                WHERE id = ?
                """,
                (conversation_id,),
            )

            return cursor.rowcount > 0


conversation_service = ConversationService()