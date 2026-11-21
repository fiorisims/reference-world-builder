"""Fast PNG writing for the analysis and model caches (numpy + zlib, no extra packages).

pygame.image.save compresses hard: about 1.3 s for a 2000 px picture, and the analysis writes dozens of
pictures.  This writes the same pixels about 4x faster, in files about 15% larger.
"""
import struct
import zlib
from pathlib import Path

import numpy as np

LEVEL = 1          # zlib level: 1 is fast, 6 (pygame's) is only ~10% smaller


def _chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def encode(array, level=LEVEL):
    """PNG bytes for an HxW (grey), HxWx3 (RGB) or HxWx4 (RGBA) uint8 array."""
    a = np.ascontiguousarray(array, np.uint8)
    if a.ndim == 2:
        a = a[..., None]
    h, w, c = a.shape
    colour = {1: 0, 3: 2, 4: 6}[c]
    # "Sub" filter on every row: each byte minus the same channel of the pixel to its left
    rows = np.empty((h, 1 + w * c), np.uint8)
    rows[:, 0] = 1
    flat = a.reshape(h, w * c)
    rows[:, 1:1 + c] = flat[:, :c]
    rows[:, 1 + c:] = flat[:, c:] - flat[:, :-c]          # uint8 arithmetic wraps, as PNG wants
    return (b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, colour, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(rows.tobytes(), level))
            + _chunk(b"IEND", b""))


def save(image, path):
    """Save a pygame Surface or a uint8 array as PNG.  Other extensions go through pygame."""
    path = Path(path)
    if not isinstance(image, np.ndarray):
        import pygame
        if path.suffix.lower() != ".png":
            pygame.image.save(image, str(path))
            return str(path)
        w, h = image.get_size()
        alpha = bool(image.get_flags() & pygame.SRCALPHA) or image.get_alpha() is not None
        mode, channels = ("RGBA", 4) if alpha else ("RGB", 3)
        image = np.frombuffer(pygame.image.tobytes(image, mode), np.uint8).reshape(h, w, channels)
    elif path.suffix.lower() != ".png":
        import pygame
        a = np.ascontiguousarray(image, np.uint8)
        mode = {3: "RGB", 4: "RGBA"}[a.shape[2]]
        pygame.image.save(pygame.image.frombuffer(a.tobytes(), (a.shape[1], a.shape[0]), mode), str(path))
        return str(path)
    path.write_bytes(encode(image))
    return str(path)
