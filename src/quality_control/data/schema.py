import sqlite3
from collections import Counter, defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from itertools import chain
from subprocess import check_call
from types import TracebackType
from typing import Iterable, Mapping
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib as mpl
from matplotlib import pyplot as plt

from file_index.bids import BIDSIndex
from compression import decompress_image


@dataclass
class Datastore(AbstractContextManager[None]):
    database: str

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
    
    def get_image_by_id(self, image_id: int) -> bytes:
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT data
            FROM image
            WHERE id = ?
            """,
            (image_id,)
        )
        result = cursor.fetchone()
        if result is None:
            raise ValueError(f"No image found with id {image_id}")
        return result[0]

    def get_images_by_direction(self, direction: str) -> list[tuple[int, str, int, bytes]]:
            """
            Retrieve all images from the image table with a specific direction.

            :param direction: The direction to filter by ('z', 'y', or 'x')
            :return: A list of tuples containing (id, direction, i, data) for each matching image
            """
            if direction not in ['z', 'y', 'x']:
                raise ValueError("Direction must be 'z', 'y', or 'x'")
            cursor = self.connection.cursor()
            cursor.execute(
                """
                SELECT id, direction, i, data
                FROM image
                WHERE direction = ?
                ORDER BY i
                """,
                (direction,)
            )
            return cursor.fetchall()
    
    def get_image_ids_by_suffix(self) -> dict[str, set[int]]:
        images_by_tags = self.get_image_ids_by_tags()
        images_by_suffix = defaultdict(set)
        
        for tags, image_ids in images_by_tags.items():
            suffix = next((value for key, value in tags if key == 'suffix'), None)
            if suffix:
                images_by_suffix[suffix].update(image_ids)
        
        return images_by_suffix

    def get_direction_i_for_images(self, image_ids: set[int]) -> list[tuple[int, str, int]]:
        return [
            (img_id, *self.get_direction_and_index(img_id))
            for img_id in image_ids
            if all(self.get_direction_and_index(img_id))
        ]
    
    def detect_outliers_for_suffix(self, suffix: str, method: str, z_score_threshold: float = 3.0) -> list[tuple[str, str, int, int]]:
        images_by_suffix = self.get_image_ids_by_suffix()
        image_ids = images_by_suffix[suffix]
        direction_i_data = self.get_direction_i_for_images(image_ids)

        grouped_data = defaultdict(lambda: defaultdict(list))
        for img_id, direction, i in direction_i_data:
            grouped_data[direction][i].append(img_id)

        outliers = []
        for direction in grouped_data:
            for i, img_ids in grouped_data[direction].items():
                # print(f"\nProcessing group {suffix}: Direction {direction}, Index {i}")
                # print(f"Number of images in this group: {len(img_ids)}")

                images = []
                max_shape = None
                for img_id in img_ids:
                    cursor = self.connection.cursor()
                    cursor.execute("SELECT data FROM image WHERE id = ?", (img_id,))
                    img_data = cursor.fetchone()[0]
                    decompressed_img = decompress_image(img_data)

                    #print(f"Image ID: {img_id}, Shape: {decompressed_img.shape}, Dimensions: {decompressed_img.ndim}")

                    images.append(decompressed_img)
                    if max_shape is None:
                        max_shape = decompressed_img.shape
                    else:
                        max_shape = tuple(max(a, b) for a, b in zip(max_shape, decompressed_img.shape))

                # print(f"Maximum shape in this group: {max_shape}")
                if method == 'intensities':
                    intensities = [np.mean(img) for img in images]
                    z_scores = np.abs(stats.zscore(intensities))
                    for idx, z_score in enumerate(z_scores):
                        if z_score > z_score_threshold:
                            outliers.append((suffix, direction, i, img_ids[idx]))
                
                elif method == 'average_image':
                    padded_images = []
                    for img in images:
                        pad_width = [(0, max_shape[j] - img.shape[j]) for j in range(len(max_shape))]
                        #print(pad_width)
                        padded_img = np.pad(img, pad_width, mode='constant', constant_values=0)
                        padded_images.append(padded_img)

                    average_image = np.mean(padded_images, axis=0)

                    differences = []
                    for img, img_id in zip(padded_images, img_ids):
                        diff = np.mean(np.abs(img - average_image))
                        differences.append((diff, img_id)) # could also use a named tuple here maybe

                    diff_values = [d[0] for d in differences] # d[0] because we have tuple of diff and id
                    z_scores = np.abs(stats.zscore(diff_values))

                    for idx, z_score in enumerate(z_scores):
                        if z_score > z_score_threshold:
                            outliers.append((suffix, direction, i, differences[idx][1]))
        return outliers

    def detect_all_outliers(self, method: str, z_score_threshold: float = 3.0) -> dict[str, list[tuple[str, str, int, int]]]:
        images_by_suffix = self.get_image_ids_by_suffix()
        all_outliers = {}

        for suffix in images_by_suffix:
            outliers = self.detect_outliers_for_suffix(suffix, method, z_score_threshold)
            all_outliers[suffix] = outliers

        return all_outliers
    def save_outlier_plot(self, outlier: tuple[str, str, int, int], output_folder: Path) -> None:
        suffix, _, _, img_id = outlier
        img_data = self.get_image_by_id(img_id)
        img_array = decompress_image(img_data)
        
        file_name = output_folder / f"{suffix}_{img_id}_anomaly.png"
        
        if suffix in ["skull_strip_report", "t1_norm_rpt", "epi_norm_rpt", "tsnr_rpt"]:
            fig, ax = plt.subplots(figsize=(20, 20))
            ax.imshow(img_array[..., 0], vmin=0, vmax=255)
            cmap = plt.get_cmap("tab20")
            for i in range(1, img_array.shape[2]):
                ax.contour(img_array[..., i], levels=[0.5], colors=[cmap(i)])
        elif suffix == "bold_conf":
            fig, axes = plt.subplots(figsize=(20, 20), nrows=img_array.shape[-1], dpi=300)
            for i, ax in enumerate(axes):
                ax.imshow(img_array[..., i], vmin=0, vmax=255)
        
        plt.tight_layout()
        fig.savefig(fname=file_name, format="png")
        plt.close(fig)