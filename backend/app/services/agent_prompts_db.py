import sqlite3
import uuid
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import contextmanager


class AgentPromptsDB:
    """SQLite database manager for agent prompts."""

    def __init__(self, db_path: str = "./conversations.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._initialize_db()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections with automatic commit/rollback."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize_db(self):
        """Create tables if they don't exist, and migrate old schema if needed."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Check if old table with notebook_id exists and migrate
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='agent_prompts'")
            table_exists = cursor.fetchone()

            if table_exists:
                # Check if old schema (has notebook_id column)
                cursor.execute("PRAGMA table_info(agent_prompts)")
                columns = {row[1] for row in cursor.fetchall()}
                if 'notebook_id' in columns:
                    # Migrate: copy data to new schema
                    cursor.execute("""
                        CREATE TABLE agent_prompts_new (
                            id TEXT PRIMARY KEY,
                            name TEXT NOT NULL,
                            description TEXT,
                            icon TEXT DEFAULT 'Bot',
                            system_prompt TEXT NOT NULL,
                            user_prompt TEXT NOT NULL,
                            template_prompt TEXT,
                            variables TEXT DEFAULT '[]',
                            created_at TIMESTAMP NOT NULL,
                            updated_at TIMESTAMP NOT NULL
                        )
                    """)
                    cursor.execute("""
                        INSERT INTO agent_prompts_new (id, name, description, icon, system_prompt, user_prompt, template_prompt, variables, created_at, updated_at)
                        SELECT id, name, description, icon, system_prompt, user_prompt, template_prompt, variables, created_at, updated_at
                        FROM agent_prompts
                    """)
                    cursor.execute("DROP TABLE agent_prompts")
                    cursor.execute("ALTER TABLE agent_prompts_new RENAME TO agent_prompts")
            else:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS agent_prompts (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT,
                        icon TEXT DEFAULT 'Bot',
                        system_prompt TEXT NOT NULL,
                        user_prompt TEXT NOT NULL,
                        template_prompt TEXT,
                        variables TEXT DEFAULT '[]',
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL
                    )
                """)

    def _serialize_json(self, data: Any) -> str:
        """Serialize data to JSON string."""
        return json.dumps(data, default=str)

    def _deserialize_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert database row to dictionary with JSON fields parsed."""
        result = dict(row)
        # Parse JSON fields
        if result.get('variables'):
            result['variables'] = json.loads(result['variables'])
        else:
            result['variables'] = []
        return result

    def create_agent_prompt(
        self,
        name: str,
        system_prompt: str,
        user_prompt: str,
        description: Optional[str] = None,
        icon: str = "Bot",
        template_prompt: Optional[str] = None,
        variables: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Create a new agent prompt.

        Args:
            name: Display name
            system_prompt: System instruction for AI behavior
            user_prompt: User instruction with placeholders
            description: Optional description
            icon: Icon name (default "Bot")
            template_prompt: Optional output template
            variables: List of variable definitions

        Returns:
            prompt_id: UUID of created agent prompt
        """
        prompt_id = str(uuid.uuid4())
        now = datetime.now()

        variables_json = self._serialize_json(variables or [])

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO agent_prompts (
                    id, name, description, icon,
                    system_prompt, user_prompt, template_prompt,
                    variables,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prompt_id, name, description, icon,
                system_prompt, user_prompt, template_prompt,
                variables_json,
                now, now
            ))

        return prompt_id

    def get_all_agent_prompts(self) -> List[Dict[str, Any]]:
        """
        Get all agent prompts.

        Returns:
            List of agent prompt dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, description, icon,
                       system_prompt, user_prompt, template_prompt,
                       variables,
                       created_at, updated_at
                FROM agent_prompts
                ORDER BY created_at ASC
            """)

            rows = cursor.fetchall()
            return [self._deserialize_row(row) for row in rows]

    def get_agent_prompt(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single agent prompt by ID.

        Args:
            prompt_id: Agent prompt UUID

        Returns:
            Agent prompt dictionary or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, description, icon,
                       system_prompt, user_prompt, template_prompt,
                       variables,
                       created_at, updated_at
                FROM agent_prompts
                WHERE id = ?
            """, (prompt_id,))

            row = cursor.fetchone()
            return self._deserialize_row(row) if row else None

    def update_agent_prompt(
        self,
        prompt_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        icon: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        template_prompt: Optional[str] = None,
        variables: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Update an agent prompt.

        Args:
            prompt_id: Agent prompt UUID
            name: New name (optional)
            description: New description (optional)
            icon: New icon (optional)
            system_prompt: New system prompt (optional)
            user_prompt: New user prompt (optional)
            template_prompt: New template prompt (optional)
            variables: New variables (optional)

        Returns:
            True if updated, False if not found
        """
        updates = []
        values = []

        if name is not None:
            updates.append("name = ?")
            values.append(name)
        if description is not None:
            updates.append("description = ?")
            values.append(description)
        if icon is not None:
            updates.append("icon = ?")
            values.append(icon)
        if system_prompt is not None:
            updates.append("system_prompt = ?")
            values.append(system_prompt)
        if user_prompt is not None:
            updates.append("user_prompt = ?")
            values.append(user_prompt)
        if template_prompt is not None:
            updates.append("template_prompt = ?")
            values.append(template_prompt)
        if variables is not None:
            updates.append("variables = ?")
            values.append(self._serialize_json(variables))

        if not updates:
            return False

        updates.append("updated_at = ?")
        values.append(datetime.now())
        values.append(prompt_id)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE agent_prompts
                SET {', '.join(updates)}
                WHERE id = ?
            """, values)

            return cursor.rowcount > 0

    def delete_agent_prompt(self, prompt_id: str) -> bool:
        """
        Delete an agent prompt.

        Args:
            prompt_id: Agent prompt UUID

        Returns:
            True if deleted, False if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM agent_prompts WHERE id = ?", (prompt_id,))
            return cursor.rowcount > 0


# Singleton instance
_agent_prompts_db_instance = None


def get_agent_prompts_db(db_path: str = "./conversations.db") -> AgentPromptsDB:
    """
    Get singleton instance of AgentPromptsDB.

    Args:
        db_path: Path to SQLite database file

    Returns:
        AgentPromptsDB instance
    """
    global _agent_prompts_db_instance
    if _agent_prompts_db_instance is None:
        _agent_prompts_db_instance = AgentPromptsDB(db_path)
    return _agent_prompts_db_instance
