import json
from pathlib import Path
from typing import Any, Iterator, MutableSequence

from tqdm.auto import tqdm

from .base import FileIndex
from .utils import split_ext


def parse(path: Path) -> Iterator[tuple[str, str]]:
    """
    Parses a BIDS-formatted filename and returns a dictionary of its tags.

    Args:
        path (Path): The path to the file to parse.

    Returns:
        dict[str, str] | None: A dictionary of the file's BIDS tags, or None if the
            file is not a valid BIDS-formatted file.
    """
    if path.is_dir():
        return  # Skip directories

    stem, extension = split_ext(path)
    if stem.startswith("."):
        return  # Skip hidden files

    keys, values = tokenize(stem)

    suffix = get_suffix(keys, values)
    yield ("suffix", suffix)

    # Build tags
    if extension:
        yield ("extension", extension)
    datatype = Path(str(path.parent)).name
    if datatype in ("anat", "func", "fmap"):
        yield ("datatype", datatype)
    for key, value in zip(keys, values, strict=False):
        if key is None:
            continue
        yield (key, value)


def get_suffix(keys: MutableSequence[str | None], values: MutableSequence[str]) -> str:
    suffixes: list[str] = list()
    while keys and keys[-1] is None:
        keys.pop(-1)
        suffixes.insert(0, values.pop(-1))
    # Merge other suffixes with their preceding tag value
    for i, (key, value) in enumerate(zip(keys, values, strict=False)):
        if i < 1:
            continue
        if key is None:
            values[i - 1] += f"_{value}"
    suffix = "_".join(suffixes)
    return suffix


def tokenize(stem: str) -> tuple[MutableSequence[str | None], MutableSequence[str]]:
    tokens = stem.split("_")
    keys: MutableSequence[str | None] = list()
    values: MutableSequence[str] = list()
    for token in tokens:
        if "-" in token:  # A bids tag
            key: str | None = token.split("-")[0]
            if key is None:
                continue
            keys.append(key)
            values.append(token[len(key) + 1 :])

        else:  # A suffix
            keys.append(None)
            values.append(token)
    return keys, values


class BIDSIndex(FileIndex):
    def put(self, root: Path) -> None:
        for path in tqdm(root.glob("**/*"), desc="indexing files", unit="files"):
            tags_iterator = parse(path)
            if tags_iterator is None:
                continue  # not a valid path

            for key, value in tags_iterator:
                self.paths_by_tags[key][value].add(path)
                self.tags_by_paths[path][key] = value

    def get_metadata(self, path: Path) -> dict[str, Any]:
        metadata: dict[str, Any] = dict()

        for metadata_path in self.get_associated_paths(path, extension=".json"):
            with metadata_path.open("r") as file:
                metadata.update(json.load(file))

        return metadata
