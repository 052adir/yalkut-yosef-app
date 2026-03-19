"""
generate_icons.py — Create placeholder PWA icons (solid color with Star of David).

Uses only Python stdlib (no Pillow needed).
Run once:  python generate_icons.py
"""

import struct
import zlib
import os

def create_png(width, height, r, g, b):
    """Create a minimal PNG with a solid color and a centered Star of David pattern."""

    def make_star_of_david(w, h, cr, cg, cb, br, bg, bb):
        """Generate raw pixel rows for a Star of David on solid background."""
        rows = []
        cx, cy = w // 2, h // 2
        size = int(min(w, h) * 0.32)

        for y in range(h):
            row = bytearray()
            row.append(0)  # PNG filter: None
            for x in range(w):
                # Check if pixel is inside the Star of David (two overlapping triangles)
                dx = x - cx
                dy = y - cy

                # Triangle pointing up (vertices at top-center, bottom-left, bottom-right)
                in_up_triangle = (
                    dy >= -size * 0.7 and
                    dy <= size * 0.5 and
                    abs(dx) <= (dy + size * 0.7) * size / (size * 1.2) * 0.95
                )

                # Triangle pointing down (vertices at bottom-center, top-left, top-right)
                in_down_triangle = (
                    dy <= size * 0.7 and
                    dy >= -size * 0.5 and
                    abs(dx) <= (-dy + size * 0.7) * size / (size * 1.2) * 0.95
                )

                if in_up_triangle or in_down_triangle:
                    row.extend([cr, cg, cb])
                else:
                    row.extend([br, bg, bb])

            rows.append(bytes(row))
        return b''.join(rows)

    # Star of David in gold (#d4a843) on theme background (#1a5276)
    raw_data = make_star_of_david(width, height, 0xd4, 0xa8, 0x43, r, g, b)

    # PNG signature
    signature = b'\x89PNG\r\n\x1a\n'

    # IHDR chunk
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    ihdr = make_chunk(b'IHDR', ihdr_data)

    # IDAT chunk (compressed pixel data)
    compressed = zlib.compress(raw_data, 9)
    idat = make_chunk(b'IDAT', compressed)

    # IEND chunk
    iend = make_chunk(b'IEND', b'')

    return signature + ihdr + idat + iend


def make_chunk(chunk_type, data):
    """Create a PNG chunk with length, type, data, and CRC."""
    chunk = chunk_type + data
    return struct.pack('>I', len(data)) + chunk + struct.pack('>I', zlib.crc32(chunk) & 0xFFFFFFFF)


def main():
    icons_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'icons')
    os.makedirs(icons_dir, exist_ok=True)

    # Background: white (#ffffff), star in gold
    r, g, b = 0xff, 0xff, 0xff

    for size in (192, 512):
        png_data = create_png(size, size, r, g, b)
        path = os.path.join(icons_dir, f'icon-{size}.png')
        with open(path, 'wb') as f:
            f.write(png_data)
        print(f"[✓] Created {path} ({len(png_data):,} bytes)")

    print("\n[✓] PWA icons generated successfully!")


if __name__ == '__main__':
    main()
