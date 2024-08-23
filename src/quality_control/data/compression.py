from contextlib import chdir
from io import BytesIO
from pathlib import Path
from shutil import which
from subprocess import DEVNULL, check_call
from tempfile import TemporaryDirectory

import numpy as np
from numpy import typing as npt
from PIL import Image
from zstandard import ZstdCompressor, ZstdDecompressor

compression_context: ZstdCompressor | None = None
decompression_context: ZstdDecompressor | None = None


def get_compression_context() -> ZstdCompressor:
    global compression_context
    if compression_context is None:
        compression_context = ZstdCompressor(level=22)
    return compression_context


def get_decompression_context() -> ZstdDecompressor:
    global decompression_context
    if decompression_context is None:
        decompression_context = ZstdDecompressor()
    return decompression_context


def compress_image(image: npt.NDArray[np.uint8]) -> bytes:
    images = [Image.fromarray(image[..., i]) for i in range(image.shape[-1])]
    io = BytesIO()
    images[0].save(
        io,
        format="png",
        save_all=True,
        append_images=images[1:],
        duration=1,
    )
    io.seek(0)
    image_bytes = io.read()

    compressed_bytes = get_compression_context().compress(image_bytes)

    apngopt = which("apngopt")
    if apngopt is not None:
        with TemporaryDirectory() as temporary_directory, chdir(temporary_directory):
            input_path = Path("input.png")
            input_path.write_bytes(image_bytes)
            output_path = Path("output.png")

            command = [apngopt, "-z2", str(input_path), str(output_path)]
            check_call(command, stdout=DEVNULL)

            apngopt_bytes = output_path.read_bytes()

        apngopt_bytes = get_compression_context().compress(apngopt_bytes)
        apngopt_image = decompress_image(apngopt_bytes)
        if apngopt_image.shape == image.shape and np.all(apngopt_image == image):
            return apngopt_bytes

    decompressed_image = decompress_image(compressed_bytes)
    if decompressed_image.shape == image.shape and np.all(decompressed_image == image):
        return compressed_bytes
    raise ValueError("Compression round-trip failed")


def decompress_image(compressed_bytes: bytes) -> npt.NDArray[np.uint8]:
    image_bytes = get_decompression_context().decompress(compressed_bytes)

    image = Image.open(BytesIO(image_bytes), formats=["png"])

    channels: list[npt.NDArray[np.uint8]] = list()
    for i in range(image.n_frames):
        image.seek(i)
        duration = int(image.info["duration"])
        for _ in range(duration):
            channels.append(np.array(image))

    return np.stack(channels, axis=-1)
