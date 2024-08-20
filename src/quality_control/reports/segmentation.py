import io
from collections import defaultdict
from copy import deepcopy
from functools import reduce
from pathlib import Path
from typing import Any, Iterable, Iterator, Union
from xml.dom.minidom import Document, Element, parse

import numpy as np
import numpy.typing as npt
from cairosvg import svg2png
from PIL import Image

from .base import color_pattern, decode_image


def parse_svg(svg_path_str: str) -> Iterator[Element]:
    document = parse(svg_path_str)
    groups = document.getElementsByTagName("g")
    for group in groups:
        group_id = group.getAttribute("id")
        if "axes" not in group_id:
            continue
        if group_id in {"axes_1", "axes_9"}:
            continue
        yield group


def create_svg_from_elements(
    elements: Iterable[Element], output_height: int, output_width: int
):
    doc = Document()

    svg = doc.createElement("svg")
    doc.appendChild(svg)

    svg.setAttribute("xmlns", "http://www.w3.org/2000/svg")
    svg.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink")
    svg.setAttribute("version", "1.1")
    svg.setAttribute("viewBox", f"0 0 {output_width} {output_height}")

    for element in elements:
        svg.appendChild(element)

    return doc.toxml()


def parse_transform(transform: str):
    transform_matrices = [np.eye(3)]

    transform_functions = transform.split(")")
    for function in transform_functions:
        function = function.strip()
        if not function:
            continue

        transform_type, data = function.split("(")
        tokens = data.split(" ")

        float_tokens = np.asarray(tokens, dtype=float)
        indices = dict(
            matrix=([0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 2, 2]),
            scale=([0, 1], [0, 1]),
            translate=([0, 1], [2, 2]),
        )[transform_type]
        transform_matrix = np.eye(3)
        transform_matrix[*indices] = float_tokens
        transform_matrices.append(transform_matrix)

    return reduce(np.matmul, transform_matrices)


def transform_to_svg(matrix) -> str:
    return (
        f"matrix({' '.join(map(str, matrix[[0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 2, 2]]))})"
    )


def svg_to_raster(
    bytestring: Union[str, io.BytesIO], output_height: int, output_width: int
) -> npt.NDArray[Any]:
    png_content = svg2png(
        bytestring=bytestring, output_height=output_height, output_width=output_width
    )

    image = np.array(Image.open(io.BytesIO(png_content)))
    image_without_alpha = image[:, :, :3]

    return image_without_alpha


def make_transform(element: Element, image: np.ndarray) -> npt.NDArray[np.float64]:
    translation_matrix = np.eye(3)
    translation_matrix[0, 2] = element.getAttribute("x")
    translation_matrix[1, 2] = element.getAttribute("y")

    if element.hasAttribute("height") and element.hasAttribute("width"):
        image_height = float(element.getAttribute("height"))
        image_width = float(element.getAttribute("width"))
    elif element.hasAttribute("height"):
        image_height = float(element.getAttribute("height"))
        image_width = image_height * image.shape[1] / image.shape[0]
    elif element.hasAttribute("width"):
        image_width = float(element.getAttribute("width"))
        image_height = image_width * image.shape[0] / image.shape[1]
    else:
        raise ValueError("No height or width found in the image element")

    scale_matrix = np.eye(3)
    scale_matrix[0, 0] = image_width / image.shape[1]
    scale_matrix[1, 1] = image_height / image.shape[0]

    image_transform = element.getAttribute("transform")
    return parse_transform(image_transform) @ translation_matrix @ scale_matrix


def prepare_path_for_raster(
    path: Element, inverse_transform: np.ndarray
) -> tuple[str, Element]:
    style = path.getAttribute("style")
    match = color_pattern.search(style)
    if match is None:
        raise ValueError(f'No color found in "{style}"')
    color = match.group("color")
    path = deepcopy(path)
    path.setAttribute("style", "fill:#008000")
    path.setAttribute("transform", transform_to_svg(inverse_transform))
    return color, path


def get_paths_from_paths_collection(
    element: Element, inverse_transform: np.ndarray
) -> Iterable[tuple[str, Element]]:
    if element.tagName == "path":
        yield prepare_path_for_raster(element, inverse_transform)
    elif element.tagName == "g":
        for path in element.getElementsByTagName("path"):
            yield prepare_path_for_raster(path, inverse_transform)
    else:
        raise ValueError(f'Unexpected tagName "{element.tagName}"')


def parse_segmentation(
    image_path: str | Path, colors: list[str]
) -> Iterator[npt.NDArray[np.uint8]]:
    found_colors: set[str] = set()
    for axes in parse_svg(str(image_path)):
        (element,) = axes.getElementsByTagName("image")

        background_image = decode_image(element)
        background_image = background_image.mean(axis=-1).astype(np.uint8)

        kwargs = dict(
            output_height=background_image.shape[0],
            output_width=background_image.shape[1],
        )
        inverse_transform = np.linalg.inv(make_transform(element, background_image))
        paths_by_color: dict[str, list[Element]] = defaultdict(list)
        for element in axes.childNodes:
            if not isinstance(element, Element):
                continue
            element_id = element.getAttribute("id")
            if ("PathCollection" not in element_id) and (
                "LineCollection" not in element_id
            ):
                continue

            for color, path in get_paths_from_paths_collection(
                element, inverse_transform
            ):
                paths_by_color[color].append(path)

        segmentation_images: list[npt.NDArray[np.uint8]] = list()
        for color in colors:
            paths = paths_by_color[color]
            if len(paths) > 0:
                found_colors.add(color)
            svg = create_svg_from_elements(paths, **kwargs)
            segmentation_image = (svg_to_raster(svg, **kwargs) != 0).any(axis=2)
            segmentation_images.append(segmentation_image)

        image = np.stack(
            [background_image, *segmentation_images], axis=2, dtype=np.uint8
        )
        # When displaying with matplotlib.pyplot images were upside down
        image = np.flip(image, axis=0)
        yield image

    missing_colors = sorted(set(colors) - found_colors)
    if missing_colors:
        raise ValueError(
            "Some of the specified colors were not found in the image: "
            f"{missing_colors}"
        )
