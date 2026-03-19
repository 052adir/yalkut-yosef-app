"""
generate_icons.py — Generate PWA icons styled as the Yalkut Yosef book cover.

Deep maroon/burgundy background with gold foil Hebrew text and ornamental borders.
Mimics the classic sefer cover: rich fabric texture, gold-stamped lettering.

Requires: pip install Pillow
Run once:  python generate_icons.py
"""

import os
import sys
import random

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("[!] Pillow is required: pip install Pillow")
    sys.exit(1)

# ── Colors matching a classic Yalkut Yosef cover ─────────────────────────────
MAROON_DARK  = (75, 18, 28)
MAROON_MID   = (90, 24, 34)
MAROON_LIGHT = (110, 32, 42)
GOLD         = (212, 168, 67)
GOLD_LIGHT   = (235, 210, 140)
GOLD_DARK    = (155, 115, 38)
GOLD_SHADOW  = (100, 72, 22)


def find_hebrew_font(size):
    """Try system Hebrew fonts, prefer bold serif for a book-cover look."""
    candidates = [
        # Windows — bold/serif Hebrew fonts
        r"C:\Windows\Fonts\davidbd.ttf",
        r"C:\Windows\Fonts\david.ttf",
        r"C:\Windows\Fonts\frank.ttf",
        r"C:\Windows\Fonts\nrkis.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf",
        # macOS
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_textured_background(draw, size):
    """Simulate book-cover fabric with subtle vertical grain + noise."""
    # Vertical grain lines (every 2-3 px, slight colour variation)
    rng = random.Random(42)  # deterministic for reproducibility
    for x in range(size):
        shade = rng.choice([MAROON_DARK, MAROON_DARK, MAROON_MID])
        draw.line([(x, 0), (x, size - 1)], fill=shade, width=1)
    # Sparse noise dots for leather/fabric feel
    for _ in range(size * size // 25):
        x = rng.randint(0, size - 1)
        y = rng.randint(0, size - 1)
        draw.point((x, y), fill=MAROON_LIGHT)


def draw_gold_rect(draw, box, width, color):
    """Draw a rectangle outline with given width."""
    x0, y0, x1, y1 = box
    for i in range(width):
        draw.rectangle([(x0 + i, y0 + i), (x1 - i, y1 - i)], outline=color)


def draw_corner_ornaments(draw, size, margin, length, width):
    """Draw small L-shaped gold corner ornaments."""
    m = margin
    cl = length
    for (cx, cy, dx, dy) in [
        (m, m, 1, 1),                        # top-right (RTL)
        (size - m, m, -1, 1),                 # top-left
        (m, size - m, 1, -1),                 # bottom-right
        (size - m, size - m, -1, -1),         # bottom-left
    ]:
        draw.line([(cx, cy), (cx + dx * cl, cy)], fill=GOLD_LIGHT, width=width)
        draw.line([(cx, cy), (cx, cy + dy * cl)], fill=GOLD_LIGHT, width=width)


def draw_ornamental_line(draw, y, left, right, size_ref):
    """Draw a horizontal gold divider with a central diamond."""
    lw = max(int(size_ref * 0.005), 1)
    draw.line([(left, y), (right, y)], fill=GOLD, width=lw)
    # Central diamond ornament
    ds = max(int(size_ref * 0.018), 3)
    cx = (left + right) // 2
    draw.polygon([
        (cx, y - ds), (cx + ds, y), (cx, y + ds), (cx - ds, y)
    ], fill=GOLD_LIGHT, outline=GOLD)


def draw_gold_text(draw, text, font, cx, cy, size_ref):
    """Draw text with shadow-then-main for a gold foil embossed effect."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = cx - tw // 2
    y = cy - th // 2

    off = max(int(size_ref * 0.005), 1)

    # Shadow layer (down-right, dark)
    draw.text((x + off, y + off), text, fill=GOLD_SHADOW, font=font)
    # Main gold layer
    draw.text((x, y), text, fill=GOLD, font=font)
    # Highlight pass (up-left, light, thinner — simulates light catch)
    # We draw a 1px offset lighter version for a subtle bevel
    draw.text((x - (off > 1 and 1 or 0), y - (off > 1 and 1 or 0)),
              text, fill=GOLD_LIGHT, font=font)
    # Redraw main on top so highlight is only an edge effect
    draw.text((x, y), text, fill=GOLD, font=font)


def create_book_cover_icon(size):
    """Create an icon resembling the Yalkut Yosef sefer cover."""
    img = Image.new('RGB', (size, size), MAROON_DARK)
    draw = ImageDraw.Draw(img)

    # 1. Textured fabric background
    draw_textured_background(draw, size)

    # 2. Double gold border frame
    outer = max(int(size * 0.055), 4)
    inner = outer + max(int(size * 0.025), 3)
    bw_outer = max(int(size * 0.008), 2)
    bw_inner = max(int(size * 0.005), 1)

    draw_gold_rect(draw, (outer, outer, size - outer - 1, size - outer - 1),
                   bw_outer, GOLD)
    draw_gold_rect(draw, (inner, inner, size - inner - 1, size - inner - 1),
                   bw_inner, GOLD_DARK)

    # 3. Corner ornaments inside inner frame
    corner_margin = inner + max(int(size * 0.025), 3)
    corner_len = max(int(size * 0.045), 5)
    corner_w = max(int(size * 0.006), 1)
    draw_corner_ornaments(draw, size, corner_margin, corner_len, corner_w)

    # 4. Ornamental divider lines
    line_left = inner + max(int(size * 0.06), 6)
    line_right = size - inner - max(int(size * 0.06), 6)
    top_line_y = int(size * 0.27)
    bot_line_y = int(size * 0.73)

    draw_ornamental_line(draw, top_line_y, line_left, line_right, size)
    draw_ornamental_line(draw, bot_line_y, line_left, line_right, size)

    # 5. Hebrew text — "ילקוט" (top) and "יוסף" (bottom)
    title_font_size = int(size * 0.16)
    sub_font_size = int(size * 0.14)
    title_font = find_hebrew_font(title_font_size)
    sub_font = find_hebrew_font(sub_font_size)

    cx = size // 2
    gap = int(size * 0.03)

    # Measure both lines to centre them vertically as a group
    b1 = draw.textbbox((0, 0), "ילקוט", font=title_font)
    b2 = draw.textbbox((0, 0), "יוסף", font=sub_font)
    h1 = b1[3] - b1[1]
    h2 = b2[3] - b2[1]
    total_h = h1 + gap + h2
    start_y = (size - total_h) // 2

    draw_gold_text(draw, "ילקוט", title_font, cx, start_y + h1 // 2, size)
    draw_gold_text(draw, "יוסף",  sub_font,   cx, start_y + h1 + gap + h2 // 2, size)

    # 6. Small decorative dash between the two words
    dash_w = max(int(size * 0.12), 10)
    dash_y = start_y + h1 + gap // 2
    dash_h = max(int(size * 0.004), 1)
    draw.rectangle(
        [(cx - dash_w // 2, dash_y - dash_h),
         (cx + dash_w // 2, dash_y + dash_h)],
        fill=GOLD_DARK
    )

    return img


def main():
    icons_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'static', 'icons')
    os.makedirs(icons_dir, exist_ok=True)

    for size in (192, 512):
        img = create_book_cover_icon(size)
        path = os.path.join(icons_dir, f'icon-{size}.png')
        img.save(path, 'PNG', optimize=True)
        file_size = os.path.getsize(path)
        print(f"[+] Created {path}  ({file_size:,} bytes, {size}x{size})")

    print("\n[+] PWA icons generated — Yalkut Yosef book cover style")
    print("    Deep maroon background + gold foil Hebrew text")


if __name__ == '__main__':
    main()
