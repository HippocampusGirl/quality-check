import sqlite3
from collections import Counter
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from itertools import chain
from subprocess import check_call
from types import TracebackType
from typing import Iterable, Mapping

from file_index.bids import BIDSIndex


@dataclass
class Datastore(AbstractContextManager[None]):
    database: str

    connection: sqlite3.Connection = field(init=False)

    def __enter__(self) -> None:
        self.connection = self.connect()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def vacuum(self) -> None:
        self.close()
        check_call(["sqlite3", self.database, "VACUUM"])

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, autocommit=False)
        with connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS string (
                    id INTEGER PRIMARY KEY,
                    data TEXT NOT NULL UNIQUE
                ) STRICT;
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS image (
                    id INTEGER PRIMARY KEY,
                    direction TEXT,
                    i INTEGER,
                    data BLOB NOT NULL
                ) STRICT;
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tag (
                    id INTEGER PRIMARY KEY,
                    key_id TEXT NOT NULL,
                    value_id TEXT NOT NULL,
                    FOREIGN KEY (key_id) REFERENCES string (id),
                    FOREIGN KEY (value_id) REFERENCES string (id)
                ) STRICT;
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS image_tag (
                    image_id INTEGER NOT NULL,
                    tag_id INTEGER NOT NULL,
                    PRIMARY KEY (image_id, tag_id),
                    FOREIGN KEY (image_id) REFERENCES image (id),
                    FOREIGN KEY (tag_id) REFERENCES tag (id)
                ) STRICT;
                """
            )
        return connection

    def close(self) -> None:
        self.connection.close()

    def set_tags_from_index(self, index: BIDSIndex) -> None:
        tag_counter = Counter(
            chain.from_iterable(tags.items() for tags in index.tags_by_paths.values())
        )
        tags = sorted(
            tag_counter.keys(), key=lambda key: tag_counter[key], reverse=True
        )
        self.set_tags(tags)

    def set_tags(self, tags: Iterable[tuple[str, str]]) -> None:
        connection = self.connection
        with connection:
            cursor = connection.cursor()

            string_counter = Counter(chain.from_iterable(tags))
            strings = sorted(
                string_counter.keys(), key=lambda key: string_counter[key], reverse=True
            )

            cursor.execute("SELECT data FROM string;")
            strings_in_datastore = list(chain.from_iterable(cursor.fetchall()))
            if len(strings_in_datastore) > 0:
                if strings_in_datastore != strings:
                    raise ValueError(
                        "Strings in datastore do not match strings in tags"
                    )
            else:
                cursor.executemany(
                    """
                    INSERT INTO string(
                        data
                    ) VALUES (
                        ?
                    );
                    """,
                    ((s,) for s in strings),
                )

            cursor.execute(
                """
                SELECT
                    key.data,
                    value.data
                FROM tag
                    LEFT JOIN string key ON tag.key_id = key.id
                    LEFT JOIN string value ON tag.value_id = value.id;
            """
            )
            tags_in_datastore = set(cursor.fetchall())
            if len(tags_in_datastore) > 0:
                if tags_in_datastore != set(tags):
                    raise ValueError("Tags in datastore do not match tags in tags")
            else:
                cursor.executemany(
                    """
                    INSERT INTO tag(
                        key_id,
                        value_id
                    ) VALUES (
                        (SELECT id FROM string WHERE data = ?),
                        (SELECT id FROM string WHERE data = ?)
                    );
                    """,
                    tags,
                )

    def get_image_ids(self, tags: Mapping[str, str]) -> list[str]:
        cursor = self.connection.cursor()

        join = " ".join(
            f"""
            LEFT JOIN tag AS {key}_tag ON {key}_tag.id = tag_id
            LEFT JOIN string AS {key}_key ON {key}_tag.key_id = {key}_key.id
            LEFT JOIN string AS {key}_value ON {key}_tag.value_id = {key}_value.id
            """
            for key in tags
        )
        where = " AND ".join(
            f"""
            {key}_key.data = ?
            AND
            {key}_value.data = ?
            """
            for key in tags
        )

        cursor.execute(
            f"""
            SELECT
                image_id
            FROM
                image_tag
                {join}
            WHERE
                {where}
            """,
            tuple(chain.from_iterable(tags.items())),
        )
        return list(chain.from_iterable(cursor.fetchall()))

    def has_image(self, tags: Mapping[str, str]) -> bool:
        return bool(self.get_image_ids(tags))

    def add_image(
        self,
        data: bytes,
        direction: str | None,
        i: int | None,
        tags: Mapping[str, str],
    ) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO image(
                direction, i, data
            ) VALUES (
                ?, ?, ?
            );
            """,
            (direction, i, data),
        )
        image_id = cursor.lastrowid
        if image_id is None:
            raise ValueError("Failed to insert image")

        cursor.executemany(
            """
            INSERT INTO image_tag(
                image_id,
                tag_id
            ) VALUES (
                ?,
                (
                    SELECT
                        id
                    FROM
                        tag
                    WHERE
                        key_id = (SELECT id FROM string WHERE data = ?)
                    AND
                        value_id = (SELECT id FROM string WHERE data = ?)
                )
            );
            """,
            ((image_id, key, value) for key, value in tags.items()),
        )
