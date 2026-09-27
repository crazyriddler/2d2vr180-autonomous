"""Generate assets/icon.png and assets/icon.ico (stylised VR headset)."""

from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]


def draw(size: int) -> Image.Image:
    s = size / 256
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([8 * s, 8 * s, 248 * s, 248 * s], radius=52 * s, fill=(33, 88, 214, 255))
    d.rounded_rectangle([34 * s, 84 * s, 222 * s, 176 * s], radius=34 * s, fill=(245, 247, 252, 255))
    d.ellipse([56 * s, 104 * s, 112 * s, 156 * s], fill=(33, 88, 214, 255))
    d.ellipse([144 * s, 104 * s, 200 * s, 156 * s], fill=(242, 94, 72, 255))
    d.polygon([(112 * s, 176 * s), (128 * s, 150 * s), (144 * s, 176 * s)], fill=(33, 88, 214, 255))
    return im


if __name__ == "__main__":
    out = REPO / "assets"
    out.mkdir(exist_ok=True)
    draw(256).save(out / "icon.png")
    draw(256).save(out / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("icons written")
