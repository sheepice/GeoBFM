"""Render paired head RGB, token magnitudes and action-query attention from HOI rollouts."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib as mpl
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CASES = ("far30_left30", "far30", "far30_right30")
LABELS = ("Left front", "Straight ahead", "Right front")
COLUMNS = ("Head RGB", "DA2 feature strength", "Body query", "Velocity query")
SIZE = 196
X = (178, 414, 650, 886)
Y = (108, 319, 530)
WHITE = "#f3f6ef"
MUTED = "#abd2ce"
DARK = "#14292d"
CMAP = mpl.colormaps["turbo"]


def overlay(rgb, field, low, high):
    smooth = Image.fromarray(field.astype("float32")).resize((112, 112), Image.Resampling.BILINEAR)
    level = np.clip((np.asarray(smooth) - low) / (high - low), 0, 1)
    color = CMAP(level)[..., :3] * 255
    return np.round(0.55 * rgb + 0.45 * color).astype(np.uint8)


def render(folder, output):
    series = []
    identity = None
    for case in CASES:
        result = json.loads((folder / case / "evaluation.json").read_text())
        current = (
            result["checkpoint_sha256"],
            result["seed"],
            result["initial_robot_qpos_sha256"],
            result["selected"]["env"],
            result["iteration"],
        )
        if identity is not None:
            assert current == identity
        identity = current
        with np.load(folder / case / "visual_features.npz") as capture:
            assert capture["queries"].tolist() == ["q_proprio", "velocity_command"]
            assert capture["env_id"] == current[3] and int(capture["control_hz"]) == 50
            tick, rgb, tok, attn = [capture[k] for k in ("tick", "rgb", "tokens", "attention")]
        assert rgb.shape == (len(tick), 112, 112, 3) and tok.shape == (len(tick), 64, 96)
        assert attn.shape == (len(tick), 8, 2, 128)
        np.testing.assert_allclose(attn.sum(-1), 1, atol=1e-6)
        magnitude = np.linalg.norm(tok.astype(np.float32), axis=-1).reshape(-1, 8, 8)
        # Preserve the original softmax denominator spanning both frames.
        weights = attn.mean(1)[..., 64:].reshape(-1, 2, 8, 8) * 128
        if series:
            np.testing.assert_array_equal(tick, series[0][0])
        series.append((tick, rgb, magnitude, weights))
    low = np.floor(min(s[2].min() for s in series) * 10) / 10
    high = np.ceil(max(s[2].max() for s in series) * 10) / 10
    gradient = np.uint8(255 * CMAP(np.linspace(0, 1, 256))[None, :, :3])
    bar = Image.fromarray(gradient).resize((112, 12))
    fonts = {size: ImageFont.load_default(size=size) for size in (16, 20, 23, 28)}
    ticks = series[0][0]

    def panel(i):
        im = Image.new("RGB", (1120, 804), DARK)
        draw = ImageDraw.Draw(im)
        draw.text((22, 14), "G1  /  Head-camera features", font=fonts[28], fill=WHITE)
        draw.text(
            (1096, 20),
            f"PPO {identity[-1]}     |     {ticks[i] / 50:.1f} s",
            anchor="ra",
            font=fonts[20],
            fill=MUTED,
        )
        for col, title in zip(X, COLUMNS, strict=True):
            draw.text((col + SIZE / 2, 72), title, anchor="ma", font=fonts[20], fill=WHITE)
        for row, (label, (_, rgb, mag, attention)) in enumerate(zip(LABELS, series, strict=True)):
            y = Y[row]
            draw.text((22, y + SIZE // 2 - 12), label, font=fonts[20], fill=WHITE)
            images = (
                rgb[i],
                overlay(rgb[i], mag[i], low, high),
                overlay(rgb[i], attention[i, 0], 0, 4),
                overlay(rgb[i], attention[i, 1], 0, 4),
            )
            for x, values in zip(X, images, strict=True):
                im.paste(Image.fromarray(values).resize((SIZE, SIZE), Image.Resampling.BILINEAR), (x, y))
        draw.text((22, 753), "Feature L2", font=fonts[20], fill=WHITE)
        draw.text((163, 755), f"{low:.1f}", anchor="ra", font=fonts[16], fill=MUTED)
        im.paste(bar, (175, 758))
        draw.text((298, 755), f"{high:.1f}", font=fonts[16], fill=MUTED)
        draw.text((490, 753), "Attention / uniform", font=fonts[20], fill=WHITE)
        draw.text((705, 755), "0", anchor="ra", font=fonts[16], fill=MUTED)
        im.paste(bar, (717, 758))
        draw.text((840, 755), "4x and above", font=fonts[16], fill=MUTED)
        return im

    output.parent.mkdir(exist_ok=True, parents=True)
    panel(0).save(output.with_suffix(".png"))
    with imageio.get_writer(
        output,
        fps=10,
        codec="libx264",
        macro_block_size=None,
        ffmpeg_params=["-threads", "2", "-crf", "17", "-movflags", "+faststart"],
    ) as writer:
        for i in range(len(ticks)):
            writer.append_data(np.asarray(panel(i)))
    print(f"{len(ticks)} visual observations per rollout; shared feature scale {low:.1f} to {high:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.captures, args.output)
