import re
from pathlib import Path
from typing import Iterator
from xml.dom.minidom import Comment, Document, Element, parse

import numpy as np
import scipy
from numpy import typing as npt

from .base import Report, color_pattern, decode_image

annotation_pattern = re.compile(
    r"max: (?P<max>[0-9\.]+) \$\\bullet\$ "
    r"mean: (?P<mean>[0-9\.]+) \$\\bullet\$ "
    r"\$\\sigma\$: (?P<sigma>[0-9\.]+)"
)

confound_estimate_keys = ["GS", "GSCSF", "GSWM", "DVARS", "FD"]


def find_comments(element: Element) -> Iterator[str]:
    stack = [element]
    while stack:
        parent = stack.pop()
        for node in parent.childNodes:
            if isinstance(node, Element):
                stack.append(node)
            if isinstance(node, Comment):
                yield node.data.strip()


def parse_path(element: Element) -> npt.NDArray[np.float64] | None:
    d = element.getAttribute("d")
    tokens = d.split()
    points_list: list[npt.NDArray[np.float64]] = []

    def add_point() -> None:
        point = np.asarray([tokens.pop(0), tokens.pop(0)], dtype=np.float64)
        if len(points_list) > 0 and np.allclose(points_list[-1], point):
            return
        points_list.append(point)

    while tokens:
        command = tokens.pop(0)
        if command in {"M", "L"}:
            add_point()
        else:
            raise ValueError(f'Unknown command "{command}"')

    if len(points_list) == 0:
        return None

    points = np.vstack(points_list)
    points[:, 1] = points[:, 1].max() - points[:, 1]  # flip y-axis
    points -= points.min(axis=0)
    return points


def scale_points(
    points: npt.NDArray[np.float64], mean: float, sigma: float, max: float
) -> npt.NDArray[np.float64]:
    y = points[:, 1]
    standard_deviation = y.std()
    if np.isclose(standard_deviation, 0):
        standard_deviation = 1  # Avoid division by zero
    y *= sigma / standard_deviation
    y += mean - y.mean()
    return points


def parse_bold_conf(image_path: str | Path) -> Iterator[Report]:
    document = parse(str(image_path))
    background_image, confound_estimates = extract_bold_conf_data(document)
    shape = background_image.shape

    channels = [background_image]
    for key in confound_estimate_keys:
        points = confound_estimates[key]

        if points is None or points.shape[0] <= 1:
            overlay = np.zeros(shape, dtype=np.uint8)
            channels.append(overlay)
            continue

        x, y = points.transpose()
        scale = (y.min(), y.max())
        if key == "FD":
            scale = (0, 5)
        y = np.interp(y, scale, (0, 255))

        f = scipy.interpolate.interp1d(x, y, kind="cubic")
        interpolated_y = f(np.linspace(x.min(), x.max(), shape[1]))
        overlay = np.broadcast_to(interpolated_y[np.newaxis, :], shape).astype(np.uint8)
        channels.append(overlay)

    yield Report(None, None, np.stack(channels, axis=2, dtype=np.uint8))


def extract_bold_conf_data(
    document: Document,
) -> tuple[npt.NDArray[np.uint8], dict[str, npt.NDArray[np.float64] | None]]:
    groups = document.getElementsByTagName("g")

    background_image: npt.NDArray[np.uint8] | None = None
    confound_estimates: dict[str, npt.NDArray[np.float64] | None] = {}
    for group in groups:
        group_id = group.getAttribute("id")
        if "axes" not in group_id:
            continue

        if group_id == "axes_7":
            (element,) = group.getElementsByTagName("image")
            background_image = decode_image(element).mean(axis=-1).astype(np.uint8)

        extracted = extract_points(group)
        if extracted is None:
            continue
        key, points = extracted
        confound_estimates[key] = points

    if background_image is None:
        raise ValueError("Could not find background image")
    return background_image, confound_estimates


def extract_points(group: Element) -> tuple[str, npt.NDArray[np.float64] | None] | None:
    comments = set(find_comments(group))
    intersection = set(confound_estimate_keys) & comments
    if not intersection:
        return None
    (key,) = intersection

    match = None
    for comment in comments:
        match = annotation_pattern.fullmatch(comment)
        if match is not None:
            break
    if match is None:
        raise ValueError(f'Could not find annotation in comments "{comments}"')

    mean, sigma, max = map(float, match.group("mean", "sigma", "max"))

    points = None
    for element in group.getElementsByTagName("path"):
        style = element.getAttribute("style")
        match = color_pattern.search(style)
        if match is None:
            continue
        color = match.group("color")
        if color == "#d3d3d3":
            continue
        points = parse_path(element)
    if points is not None:
        points = scale_points(points, mean, sigma, max)
    return key, points
