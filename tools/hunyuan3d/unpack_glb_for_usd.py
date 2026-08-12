#!/usr/bin/env python3
"""Unpack embedded GLB images/buffers without changing its glTF material graph."""

from __future__ import annotations

import argparse
from pathlib import Path

from pygltflib import BufferFormat, GLTF2, ImageFormat


def unpack(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    gltf = GLTF2.load(source)
    gltf.convert_images(ImageFormat.FILE, path=output.parent, override=True)
    gltf._path = output.parent
    gltf.convert_buffers(BufferFormat.BINFILE, override=True)
    gltf.save(output)
    if not output.is_file():
        raise RuntimeError(f"glTF output was not created: {output}")
    for image in gltf.images:
        if not image.uri or image.uri.startswith("data:") or not (output.parent / image.uri).is_file():
            raise RuntimeError(f"Image was not externalized correctly: {image.uri!r}")
    for buffer in gltf.buffers:
        if not buffer.uri or not (output.parent / buffer.uri).is_file():
            raise RuntimeError(f"Buffer was not externalized correctly: {buffer.uri!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    unpack(args.source.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
