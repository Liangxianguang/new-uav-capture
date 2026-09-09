"""Render a dependency-light 3-D perspective animation from a saved trajectory.

The renderer deliberately uses Pillow rather than Matplotlib's 3-D backend.
This keeps PNG/GIF/FFmpeg MP4 generation stable in the Windows conda runtime
used by the project while preserving obstacle volumes and altitude cues.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def find_ffmpeg() -> str | None:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    for candidate in (Path(sys.prefix) / "Library" / "bin" / "ffmpeg.exe", Path(sys.prefix) / "bin" / "ffmpeg"):
        if candidate.is_file():
            return str(candidate)
    try:
        import imageio_ffmpeg

        bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if bundled.is_file():
            return str(bundled)
    except (ImportError, RuntimeError, OSError):
        pass
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--tail-length", type=int, default=0, help="0 keeps the complete trajectory tail.")
    parser.add_argument("--freeze-seconds", type=float, default=1.75)
    return parser.parse_args()


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _project(points: np.ndarray, extent: float, world_height: float) -> tuple[np.ndarray, np.ndarray]:
    """Project world coordinates through a fixed, lightly overhead camera."""
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    azimuth = np.deg2rad(-48.0)
    elevation = np.deg2rad(48.0)
    x_rot = np.cos(azimuth) * values[:, 0] - np.sin(azimuth) * values[:, 1]
    depth_axis = np.sin(azimuth) * values[:, 0] + np.cos(azimuth) * values[:, 1]
    # Positive world z must map upward on screen, where smaller y is higher.
    y_rot = np.cos(elevation) * depth_axis + np.sin(elevation) * values[:, 2]
    depth = -np.sin(elevation) * depth_axis + np.cos(elevation) * values[:, 2]
    camera_distance = 34.0
    scale = 22.0 * min(1.0, 10.0 / max(extent, 1e-6))
    perspective = camera_distance / np.maximum(6.0, camera_distance + depth)
    screen_x = 640.0 + x_rot * scale * perspective
    screen_y = 454.0 - y_rot * scale * perspective
    return np.column_stack((screen_x, screen_y)), depth


def _line(draw: ImageDraw.ImageDraw, points: np.ndarray, fill: tuple[int, int, int, int], width: int = 1) -> None:
    if len(points) >= 2:
        draw.line([tuple(map(float, point)) for point in points], fill=fill, width=width, joint="curve")


def _draw_grid(draw: ImageDraw.ImageDraw, extent: float, world_height: float) -> None:
    grid = (188, 201, 213, 150)
    for coordinate in np.linspace(-extent, extent, 9):
        projected, _ = _project(np.array([[coordinate, -extent, 0.0], [coordinate, extent, 0.0]]), extent, world_height)
        _line(draw, projected, grid)
        projected, _ = _project(np.array([[-extent, coordinate, 0.0], [extent, coordinate, 0.0]]), extent, world_height)
        _line(draw, projected, grid)
    outline = np.array(
        [
            [-extent, -extent, 0.0],
            [extent, -extent, 0.0],
            [extent, extent, 0.0],
            [-extent, extent, 0.0],
            [-extent, -extent, 0.0],
        ]
    )
    projected, _ = _project(outline, extent, world_height)
    _line(draw, projected, (142, 160, 177, 190), width=2)


def _draw_prism(
    draw: ImageDraw.ImageDraw,
    center: np.ndarray,
    half_extent: np.ndarray,
    height: float,
    extent: float,
    world_height: float,
    color: tuple[int, int, int],
) -> None:
    half_x, half_y = map(float, half_extent)
    x, y = map(float, center)
    ground = np.array([[x - half_x, y - half_y, 0.0], [x + half_x, y - half_y, 0.0], [x + half_x, y + half_y, 0.0], [x - half_x, y + half_y, 0.0]])
    top = ground + np.array([0.0, 0.0, height])
    vertices = np.vstack((ground, top))
    projected, depth = _project(vertices, extent, world_height)
    faces = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)]
    for face in sorted(faces, key=lambda indices: float(np.mean(depth[list(indices)])), reverse=True):
        polygon = [tuple(map(float, projected[index])) for index in face]
        shade = 0.76 + 0.18 * (face == (4, 5, 6, 7))
        fill = tuple(int(channel * shade) for channel in color) + (118,)
        outline = tuple(max(0, int(channel * 0.72)) for channel in color) + (180,)
        draw.polygon(polygon, fill=fill, outline=outline)


def _draw_cylinder(
    draw: ImageDraw.ImageDraw,
    center: np.ndarray,
    radius: float,
    height: float,
    extent: float,
    world_height: float,
    color: tuple[int, int, int],
) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 28, endpoint=False)
    ring_xy = np.column_stack((center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)))
    bottom = np.column_stack((ring_xy, np.zeros(len(theta))))
    top = bottom + np.array([0.0, 0.0, height])
    vertices = np.vstack((bottom, top))
    projected, depth = _project(vertices, extent, world_height)
    top_projected = projected[len(theta) :]
    outline = tuple(max(0, int(channel * 0.72)) for channel in color) + (185,)
    faces = [
        (index, (index + 1) % len(theta), (index + 1) % len(theta) + len(theta), index + len(theta))
        for index in range(len(theta))
    ]
    for face in sorted(faces, key=lambda indices: float(np.mean(depth[list(indices)])), reverse=True):
        polygon = [tuple(map(float, projected[index])) for index in face]
        draw.polygon(polygon, fill=color + (82,), outline=outline)
    draw.polygon([tuple(map(float, point)) for point in top_projected], fill=color + (132,), outline=outline)
    _line(draw, np.vstack((top_projected, top_projected[0])), outline, width=2)


def _draw_obstacles(draw: ImageDraw.ImageDraw, data: dict[str, Any]) -> None:
    centers = data["centers"]
    radii = data["radii"]
    heights = data["heights"]
    shapes = data["shapes"]
    half_extents = data["half_extents"]
    extent = data["extent"]
    world_height = data["world_height"]
    colors = {"cylinder": (76, 129, 186), "box": (65, 169, 162), "wall": (226, 119, 104)}
    ordering: list[tuple[float, int]] = []
    for index, center in enumerate(centers):
        _, depth = _project(np.array([[center[0], center[1], 0.0]]), extent, world_height)
        ordering.append((float(depth[0]), index))
    for _, index in sorted(ordering):
        shape = str(shapes[index])
        color = colors.get(shape, (92, 114, 139))
        if shape == "cylinder":
            _draw_cylinder(draw, centers[index], float(radii[index]), float(heights[index]), extent, world_height, color)
        else:
            _draw_prism(draw, centers[index], half_extents[index], float(heights[index]), extent, world_height, color)


def _heading_on_screen(
    trajectory: np.ndarray,
    frame_index: int,
    extent: float,
    world_height: float,
) -> np.ndarray:
    """Infer an icon heading from the nearest non-stationary trajectory segment."""
    current = np.asarray(trajectory[frame_index], dtype=np.float64)
    for offset in range(1, 6):
        for candidate_index in (frame_index + offset, frame_index - offset):
            if 0 <= candidate_index < len(trajectory):
                candidate = np.asarray(trajectory[candidate_index], dtype=np.float64)
                if candidate_index < frame_index:
                    delta = current - candidate
                else:
                    delta = candidate - current
                if float(np.linalg.norm(delta)) > 1e-6:
                    projected, _ = _project(np.vstack((current, current + delta)), extent, world_height)
                    direction = projected[1] - projected[0]
                    magnitude = float(np.linalg.norm(direction))
                    if magnitude > 1e-6:
                        return direction / magnitude
    return np.array([0.0, -1.0], dtype=np.float64)


def _draw_drone(
    draw: ImageDraw.ImageDraw,
    point: np.ndarray,
    heading: np.ndarray,
    color: tuple[int, int, int],
    label: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    """Draw a compact aircraft icon whose nose points along the trajectory."""
    forward = np.asarray(heading, dtype=np.float64)
    forward /= max(float(np.linalg.norm(forward)), 1e-6)
    lateral = np.array([-forward[1], forward[0]], dtype=np.float64)
    nose = point + 12.0 * forward
    tail = point - 9.0 * forward
    body = [
        tuple(nose),
        tuple(point + 6.0 * lateral),
        tuple(tail + 2.0 * lateral),
        tuple(tail),
        tuple(tail - 2.0 * lateral),
        tuple(point - 6.0 * lateral),
    ]
    draw.polygon(body, fill=color + (255,), outline=(43, 61, 76, 230))
    wing_start = point - 1.5 * forward
    _line(draw, np.vstack((wing_start - 9.0 * lateral, wing_start + 9.0 * lateral)), (43, 61, 76, 230), width=2)
    _line(draw, np.vstack((tail - 3.0 * lateral, tail + 3.0 * lateral)), (43, 61, 76, 220), width=1)
    label_point = point + 11.0 * lateral - 8.0 * forward
    draw.text(tuple(label_point), label, font=font, fill=(37, 54, 69, 255), stroke_width=1, stroke_fill=(255, 255, 255, 235))


def _status_text(safe_capture: bool, final_frame: bool, termination_reason: str | None) -> tuple[str, tuple[int, int, int]]:
    if final_frame and safe_capture:
        return "CAPTURE CONFIRMED", (24, 132, 116)
    if final_frame and termination_reason == "timeout":
        return "TIMEOUT", (183, 104, 31)
    if final_frame and termination_reason:
        return "SAFETY TERMINATION", (187, 67, 68)
    return "INTERCEPTION IN PROGRESS", (57, 101, 145)


def _draw_frame(
    data: dict[str, Any],
    frame_index: int,
    tail_length: int,
    safe_capture: bool,
    final_frame: bool,
    termination_reason: str | None = None,
) -> Image.Image:
    image = Image.new("RGBA", (1280, 760), (248, 250, 252, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    title_font, body_font, small_font = _font(23, True), _font(15), _font(12)
    draw.rectangle((0, 0, 1280, 78), fill=(255, 255, 255, 255))
    draw.line((0, 78, 1280, 78), fill=(209, 219, 228, 255), width=1)
    draw.text((34, 18), "MULTI-UAV INTERCEPTION REPLAY", font=title_font, fill=(35, 54, 69, 255))
    draw.text((36, 50), "FIXED-CAMERA 3-D TRAJECTORY", font=small_font, fill=(101, 120, 137, 255))
    status, status_color = _status_text(safe_capture, final_frame, termination_reason)
    draw.rectangle((936, 20, 1246, 56), fill=(255, 255, 255, 255), outline=status_color + (255,), width=2)
    draw.text((956, 30), status, font=body_font, fill=status_color + (255,))

    _draw_grid(draw, data["extent"], data["world_height"])
    _draw_obstacles(draw, data)

    defenders = data["defenders"]
    target = data["target"]
    start = 0 if tail_length == 0 else max(0, frame_index - tail_length)
    colors = ((41, 123, 190), (224, 128, 49), (70, 151, 100), (135, 95, 174))
    for defender_index in range(defenders.shape[1]):
        projected, _ = _project(defenders[start : frame_index + 1, defender_index], data["extent"], data["world_height"])
        _line(draw, projected, colors[defender_index % len(colors)] + (205,), width=3)
    target_projected, _ = _project(target[start : frame_index + 1], data["extent"], data["world_height"])
    _line(draw, target_projected, (205, 61, 72, 235), width=4)

    target_position = target[frame_index]
    target_screen, _ = _project(target_position[None, :], data["extent"], data["world_height"])
    radius_points, _ = _project(
        np.array([target_position + [data["capture_radius"], 0.0, 0.0], target_position + [0.0, data["capture_radius"], 0.0]]),
        data["extent"],
        data["world_height"],
    )
    capture_px = max(12, int(np.mean(np.linalg.norm(radius_points - target_screen[0], axis=1))))
    capture_color = (205, 61, 72, 230)
    point = target_screen[0]
    draw.ellipse((point[0] - capture_px, point[1] - capture_px, point[0] + capture_px, point[1] + capture_px), outline=capture_color, width=3)
    draw.ellipse((point[0] - 7, point[1] - 7, point[0] + 7, point[1] + 7), fill=(205, 61, 72, 255), outline=(255, 255, 255, 255), width=2)

    for defender_index, position in enumerate(defenders[frame_index]):
        projected, _ = _project(position[None, :], data["extent"], data["world_height"])
        point = projected[0]
        color = colors[defender_index % len(colors)]
        heading = _heading_on_screen(defenders[:, defender_index], frame_index, data["extent"], data["world_height"])
        _draw_drone(draw, point, heading, color, f"D{defender_index + 1}", small_font)

    distances = np.linalg.norm(defenders[frame_index] - target_position[None, :], axis=1)
    draw.rectangle((32, 680, 1248, 730), fill=(255, 255, 255, 238), outline=(209, 219, 228, 255), width=1)
    draw.text((54, 693), f"TIME  {frame_index * 0.1:05.1f} s", font=body_font, fill=(35, 54, 69, 255))
    draw.text((318, 693), f"NEAREST DEFENDER  {float(np.min(distances)):.2f} m", font=body_font, fill=(35, 54, 69, 255))
    draw.text((710, 693), f"CAPTURE RADIUS  {data['capture_radius']:.2f} m", font=body_font, fill=(35, 54, 69, 255))
    draw.text((1010, 693), "TARGET", font=body_font, fill=(205, 61, 72, 255))
    return image.convert("RGB")


def _load_scene(trajectory_path: Path) -> dict[str, Any]:
    raw = np.load(trajectory_path)
    centers = np.asarray(raw["obstacle_centers_xy"], dtype=np.float64)
    radii = np.asarray(raw["obstacle_radii"], dtype=np.float64)
    return {
        "defenders": np.asarray(raw["defender_positions"], dtype=np.float64),
        "target": np.asarray(raw["target_positions"], dtype=np.float64),
        "centers": centers,
        "radii": radii,
        "heights": np.asarray(raw["obstacle_heights"], dtype=np.float64),
        "shapes": np.asarray(raw["obstacle_shapes"]).astype(str) if "obstacle_shapes" in raw.files else np.full(len(centers), "cylinder", dtype="U16"),
        "half_extents": np.asarray(raw["obstacle_half_extents_xy"], dtype=np.float64) if "obstacle_half_extents_xy" in raw.files else np.repeat(radii[:, None], 2, axis=1),
        "extent": float(raw["world_half_extent"]),
        "world_height": float(raw["world_height"]),
        "capture_radius": float(raw["capture_radius"]),
    }


def render_static_perspective(trajectory_path: Path, output_path: Path, result: dict[str, Any]) -> None:
    """Write the final-frame perspective used by legacy replay output."""
    scene = _load_scene(trajectory_path)
    image = _draw_frame(
        scene,
        len(scene["target"]) - 1,
        0,
        bool(result.get("safe_capture_success")),
        True,
        str(result.get("termination_reason", "")),
    )
    image.save(output_path)


def _write_mp4(frames: list[Image.Image], output_path: Path, fps: int) -> None:
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("FFmpeg was not found.")
    png_stream = bytearray()
    for frame in frames:
        buffer = io.BytesIO()
        frame.save(buffer, format="PNG")
        png_stream.extend(buffer.getvalue())
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "image2pipe", "-vcodec", "png", "-r", str(fps), "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(output_path)],
        input=bytes(png_stream),
        check=True,
    )


def render_animation(trajectory_path: Path, result_path: Path, output_dir: Path, fps: int, frame_stride: int, tail_length: int, freeze_seconds: float) -> dict[str, Any]:
    if fps <= 0 or frame_stride <= 0 or tail_length < 0 or freeze_seconds < 0:
        raise ValueError("fps, frame-stride, tail-length and freeze-seconds must be non-negative as appropriate")
    scene = _load_scene(trajectory_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    safe_capture = bool(result.get("safe_capture_success", False))
    indices = np.arange(0, len(scene["target"]), frame_stride, dtype=np.int64)
    if indices[-1] != len(scene["target"]) - 1:
        indices = np.append(indices, len(scene["target"]) - 1)
    frames = [
        _draw_frame(
            scene,
            int(index),
            tail_length,
            safe_capture,
            int(index) == len(scene["target"]) - 1,
            str(result.get("termination_reason", "")),
        )
        for index in indices
    ]
    if safe_capture and freeze_seconds > 0:
        frames.extend([frames[-1].copy() for _ in range(max(1, int(round(freeze_seconds * fps))))])
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / "capture_3d.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=max(1, int(1000 / fps)), loop=0, optimize=False)
    mp4_path = output_dir / "capture_3d.mp4"
    _write_mp4(frames, mp4_path, fps)
    final_png = output_dir / "capture_3d_final.png"
    frames[-1].save(final_png)
    distances = np.linalg.norm(scene["defenders"][-1] - scene["target"][-1][None, :], axis=1)
    return {
        "gif": str(gif_path),
        "mp4": str(mp4_path),
        "final_png": str(final_png),
        "frames": len(frames),
        "fps": fps,
        "capture_radius_m": scene["capture_radius"],
        "final_nearest_distance_m": float(np.min(distances)),
        "safe_capture_success": safe_capture,
    }


def main() -> None:
    args = parse_args()
    trajectory = args.trajectory.resolve()
    result = args.result.resolve()
    output_dir = args.output_dir.resolve()
    if not trajectory.is_file() or not result.is_file():
        raise FileNotFoundError("trajectory or result JSON does not exist")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    media = render_animation(trajectory, result, output_dir, args.fps, args.frame_stride, args.tail_length, args.freeze_seconds)
    (output_dir / "media.json").write_text(json.dumps(media, indent=2), encoding="utf-8")
    print(json.dumps(media, indent=2), flush=True)


if __name__ == "__main__":
    main()
