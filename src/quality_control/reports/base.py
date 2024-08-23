import io
import re
from typing import NamedTuple
from urllib.request import urlopen
from xml.dom.minidom import Element

import numpy as np
import numpy.typing as npt
from PIL import Image

xlink_ns = "http://www.w3.org/1999/xlink"


def decode_image(element: Element) -> npt.NDArray[np.uint8]:
    href = element.getAttributeNS(xlink_ns, "href")
    with urlopen(href) as response:
        data = response.read()
    image = Image.open(io.BytesIO(data))
    return np.array(image)[..., :3]


color_pattern = re.compile(r"stroke:\s*(?P<color>[^;]+)")


class Report(NamedTuple):
    direction: str | None
    axes_index: int | None
    image: npt.NDArray[np.uint8]
