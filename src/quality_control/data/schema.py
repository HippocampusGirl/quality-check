import sqlite3
from collections import Counter, defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from itertools import chain
from types import TracebackType
from typing import Iterable, Mapping

from file_index.bids import BIDSIndex


@dataclass
class Datastore(AbstractContextManager[None]):
    database_uri: str

    connection: sqlite3.Connection | None = None

    def __enter__(self) -> None:
        self.connection = self.connect()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_uri, autocommit=False, uri=True)
        with connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS string (
                    id INTEGER PRIMARY KEY,
                    data TEXT NOT NULL UNIQUE
                ) STRICT
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS image (
                    id INTEGER PRIMARY KEY,
                    direction TEXT,
                    i INTEGER,
                    data BLOB NOT NULL
                ) STRICT
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
                ) STRICT
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
                ) STRICT
                """
            )
        return connection

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

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
        if connection is None:
            raise ValueError("Connection is not open")
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
                    )
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
                    LEFT JOIN string value ON tag.value_id = value.id
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
                    )
                    """,
                    tags,
                )

    def get_image_ids(self, tags: Mapping[str, str]) -> set[int]:
        connection = self.connection
        if connection is None:
            raise ValueError("Connection is not open")
        cursor = connection.cursor()

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
        return set(chain.from_iterable(cursor.fetchall()))

    def has_image(self, tags: Mapping[str, str]) -> bool:
        return bool(self.get_image_ids(tags))

    def get_image_ids_by_tags(self) -> dict[frozenset[tuple[str, str]], set[int]]:
        connection = self.connection
        if connection is None:
            raise ValueError("Connection is not open")
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                image_id, key.data, value.data
            FROM image_tag
                LEFT JOIN tag ON tag.id = tag_id
                LEFT JOIN string key ON tag.key_id = key.id
                LEFT JOIN string value ON tag.value_id = value.id
            """
        )
        tags_by_image: dict[int, set[tuple[str, str]]] = defaultdict(set)
        for image_id, key, value in cursor.fetchall():
            tags_by_image[image_id].add((key, value))

        images_by_tags: dict[frozenset[tuple[str, str]], set[int]] = defaultdict(set)
        for image_id, image_tags in tags_by_image.items():
            images_by_tags[frozenset(image_tags)].add(image_id)
        return images_by_tags

    def get_image(self, image_id: int) -> bytes:
        connection = self.connection
        if connection is None:
            raise ValueError("Connection is not open")
        cursor = connection.cursor()

        cursor.execute("SELECT data FROM image WHERE id = ?", (image_id,))
        (data,) = cursor.fetchone()
        if not isinstance(data, bytes):
            raise ValueError("Data is not bytes")
        return data

    def get_direction_and_index(self, image_id: int) -> tuple[str | None, int | None]:
        connection = self.connection
        if connection is None:
            raise ValueError("Connection is not open")
        cursor = connection.cursor()

        cursor.execute("SELECT direction, i FROM image WHERE id = ?", (image_id,))
        direction, i = cursor.fetchone()
        return direction, i

    def add_image(
        self,
        data: bytes,
        direction: str | None,
        i: int | None,
        tags: Mapping[str, str],
    ) -> None:
        connection = self.connection
        if connection is None:
            raise ValueError("Connection is not open")
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO image(
                direction, i, data
            ) VALUES (
                ?, ?, ?
            )
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
            )
            """,
            ((image_id, key, value) for key, value in tags.items()),
        )

    def remove_image(self, image_id: int) -> None:
        connection = self.connection
        if connection is None:
            raise ValueError("Connection is not open")
        cursor = connection.cursor()

        cursor.execute("DELETE FROM image WHERE id = ?", (image_id,))
        cursor.execute("DELETE FROM image_tag WHERE image_id = ?", (image_id,))
