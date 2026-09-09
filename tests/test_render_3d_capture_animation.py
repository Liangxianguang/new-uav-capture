from __future__ import annotations

import numpy as np

from scripts.render_3d_capture_animation import _draw_frame, _project


def test_projection_maps_positive_height_upward_on_screen() -> None:
    low, _ = _project(np.array([[0.0, 0.0, 0.0]], dtype=np.float64), extent=10.0, world_height=10.0)
    high, _ = _project(np.array([[0.0, 0.0, 4.0]], dtype=np.float64), extent=10.0, world_height=10.0)

    assert high[0, 1] < low[0, 1]


def test_publication_renderer_draws_a_light_rgb_frame() -> None:
    data = {
        "defenders": np.array(
            [
                [[-1.0, 0.0, 1.0], [0.0, 1.0, 1.5]],
                [[-0.5, 0.0, 2.0], [0.5, 1.0, 1.5]],
            ],
            dtype=np.float64,
        ),
        "target": np.array([[1.0, 0.0, 1.0], [1.2, 0.0, 1.2]], dtype=np.float64),
        "centers": np.array([[0.0, 0.0]], dtype=np.float64),
        "radii": np.array([0.5], dtype=np.float64),
        "heights": np.array([3.0], dtype=np.float64),
        "shapes": np.array(["cylinder"]),
        "half_extents": np.array([[0.5, 0.5]], dtype=np.float64),
        "extent": 5.0,
        "world_height": 5.0,
        "capture_radius": 0.8,
    }

    frame = _draw_frame(data, frame_index=1, tail_length=0, safe_capture=True, final_frame=True)

    assert frame.mode == "RGB"
    assert frame.size == (1280, 760)
    assert frame.getpixel((0, 759)) == (248, 250, 252)
