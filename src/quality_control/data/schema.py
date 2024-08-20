import sqlite3
from collections import Counter
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from itertools import chain
from subprocess import check_call
from types import TracebackType
from typing import Iterable, Mapping

from file_index.bids import BIDSIndex
from zstandard import ZstdCompressor


@dataclass
class Datastore(AbstractContextManager):
    database: str

    connection: sqlite3.Connection = field(init=False)

    compression_context = ZstdCompressor(level=22)

    def __enter__(self) -> None:
        self.connection = self.connect()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self.close()

    def vacuum(self) -> None:
        self.close()
        check_call(["sqlite3", self.database, "VACUUM"])

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database)
        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS image (
                id INTEGER PRIMARY KEY,
                data BLOB NOT NULL,
                i INTEGER NOT NULL
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS string (
                id INTEGER PRIMARY KEY,
                data TEXT NOT NULL
            );
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
            );
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
            );
            """
        )
        connection.commit()
        return connection

    def close(self):
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
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT data from string;
        """
        )
        strings_in_datastore = list(chain.from_iterable(cursor.fetchall()))

        string_counter = Counter(chain.from_iterable(tags))
        for s in strings_in_datastore:
            if s in string_counter:
                del string_counter[s]

        strings = sorted(
            string_counter.keys(), key=lambda key: string_counter[key], reverse=True
        )
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
        tags_in_datastore = cursor.fetchall()
        for t in tags_in_datastore:
            if t in tags:
                tags.remove(t)

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
        self.connection.commit()

    def add_image(self, data: bytes, i: int, tags: Mapping[str, str]) -> int:
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO image(
                data, i
            ) VALUES (
                ?, ?
            );
            """,
            (self.compression_context.compress(data), i),
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
        self.connection.commit()
