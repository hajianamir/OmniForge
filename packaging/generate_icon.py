"""Generate enterprise multi-resolution Windows .ico and PNG assets for OmniForge."""

from __future__ import annotations

from pathlib import Path


def generate_omniforge_icon(output_dir: Path | None = None) -> tuple[Path, Path]:
    """Programmatically render anti-aliased OmniForge logo PNG and multi-size Windows ICO.

    Args:
        output_dir: Destination directory for ``omniforge.ico`` and ``omniforge_logo.png``.
            Defaults to ``packaging/assets``.

    Returns:
        Tuple of ``(ico_path, png_path)``.
    """
    from PIL import Image, ImageDraw

    assets_dir = output_dir or (Path(__file__).resolve().parent / "assets")
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Render at 1024x1024 and downsample with LANCZOS for crisp anti-aliasing
    size = 1024
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background rounded container (deep enterprise slate-navy)
    margin = 48
    radius = 196
    draw.rounded_rectangle(
        [margin, margin, size - margin, size - margin],
        radius=radius,
        fill=(13, 25, 48, 255),
        outline=(56, 189, 248, 255),
        width=24,
    )

    # Subtle inner border highlight
    inner = 92
    draw.rounded_rectangle(
        [inner, inner, size - inner, size - inner],
        radius=152,
        outline=(30, 58, 95, 255),
        width=8,
    )

    # Rising analytics bars (gradient tones from teal to electric cyan)
    bars = [
        (210, 590, 310, 810, (14, 165, 233, 255)),
        (360, 480, 460, 810, (6, 182, 212, 255)),
        (510, 370, 610, 810, (20, 184, 166, 255)),
        (660, 260, 760, 810, (52, 211, 153, 255)),
    ]
    for x0, y0, x1, y1, color in bars:
        draw.rounded_rectangle([x0, y0, x1, y1], radius=24, fill=color)

    # Predictive trend polyline & decision intelligence nodes above bars
    nodes = [
        (260, 470),
        (410, 385),
        (560, 290),
        (710, 195),
    ]
    draw.line(nodes, fill=(248, 250, 252, 255), width=22, joint="curve")

    for cx, cy in nodes:
        r_outer = 38
        r_inner = 18
        draw.ellipse(
            [cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer],
            fill=(13, 25, 48, 255),
            outline=(56, 189, 248, 255),
            width=14,
        )
        draw.ellipse(
            [cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner],
            fill=(248, 250, 252, 255),
        )

    # Baseline accent bar
    draw.rounded_rectangle([180, 825, 844, 855], radius=14, fill=(148, 163, 184, 220))

    png_path = assets_dir / "omniforge_logo.png"
    ico_path = assets_dir / "omniforge.ico"

    img_256 = img.resize((256, 256), Image.Resampling.LANCZOS)
    img_256.save(png_path, format="PNG")

    icon_sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    img.save(ico_path, format="ICO", sizes=icon_sizes)
    return ico_path, png_path


def main() -> None:
    """CLI entry point to generate OmniForge icon assets."""
    ico_path, png_path = generate_omniforge_icon()
    print(f"Generated Windows icon : {ico_path}")
    print(f"Generated logo PNG     : {png_path}")


if __name__ == "__main__":
    main()
