"""
Visualization utilities for quantum circuits.

This module handles the drawing of circuit diagrams using Matplotlib.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

if TYPE_CHECKING:
    from .circuit import QuantumCircuit


def draw_circuit(
    qc: QuantumCircuit,
    scheduling: str = "asap",
    font_color: str = "black",
    show_edges: bool = False,
    gate_colors: tuple[str] = (
        "#EFBB04",  # Pauli
        "#0DB8B8",  # H/S
        "#4A90E2",  # CNOT
        "#9B59B6",  # CZ
        "#878CED",  # Other
    ),
    save_path: Optional[str] = None,
    dpi: int = 300,
    show_title: bool = False,
    show_info: bool = False,
    show_image: bool = True,
) -> None:
    """
    Backend implementation for QuantumCircuit.draw().

    See `circuit.QuantumCircuit.draw` for full parameter documentation.
    """
    base_spacing: float = (
        1.5  # center‑to‑center gap on both wires & CNOTs (default=1.5)
    )
    gate_w: float = 1.0  # 1‑qudit gate width (default=1.0)
    gate_h: float = (
        1.0  # 1‑qudit gate height (also sets CNOT dot/circle scale) (default=1.0)
    )

    if not show_edges:
        edgecolor = None
    else:
        edgecolor = "black"

    # pick scheduling method
    if scheduling == "asap":
        cols = qc._asap_schedule()
    elif scheduling == "asap_packed":
        cols = qc._asap_schedule_packed()
    elif scheduling == "logical":
        cols = qc._logical_schedule()
    else:
        raise ValueError(f"Unknown scheduling '{scheduling}' – use 'asap' or 'logical'")
    ncols = len(cols)

    # Global style settings
    rcParams["path.snap"] = False
    rcParams["text.antialiased"] = True
    rcParams["lines.antialiased"] = True
    rcParams["patch.antialiased"] = True

    # uniform x‑positions for every column
    x_wire_start = base_spacing
    x0 = x_wire_start + gate_w
    xs = [x0 + i * base_spacing for i in range(ncols)]

    # figure setup
    total_width = xs[-1] + gate_w if xs else x_wire_start + gate_w
    total_height = qc.n * base_spacing
    fig, ax = plt.subplots(figsize=(min(total_width, 40), min(total_height, 20)))
    ax.set_aspect("equal", adjustable="box")

    window_title = qc.name if qc.name else "QuantumCircuit"
    fig.canvas.manager.set_window_title(window_title)

    # lists to store text & line artists that need dynamic resizing
    text_artists = []
    line_artists = []

    if show_info:
        info_str = (
            f"name='{qc.name}', n={qc.n}, d={qc.d}, size={qc.size}, depth={qc.depth}"
        )
        title_obj = ax.set_title(info_str, color=font_color)
        text_artists.append({"artist": title_obj, "type": "title"})
    elif show_title and qc.name:
        title_obj = ax.set_title(qc.name, color=font_color)
        text_artists.append({"artist": title_obj, "type": "title"})

    # draw horizontal wires and labels
    ys = {i: (qc.n - 1 - i) * base_spacing for i in range(qc.n)}
    x_wire_end_full = xs[-1] + gate_w if xs else x_wire_start + gate_w

    wire_end_x = {i: x_wire_end_full for i in range(qc.n)}
    for col_idx, col in enumerate(cols):
        for gate, qudits, dagger in col:
            if gate.name == "M":
                qudit_idx = qudits[0]
                wire_end_x[qudit_idx] = xs[col_idx]

    for i in range(qc.n):
        current_x_end = wire_end_x[i]
        (h_line,) = ax.plot(
            [x_wire_start, current_x_end],
            [ys[i], ys[i]],
            color="black",
            linewidth=1.5,
            zorder=0,
            solid_capstyle="butt",
        )
        line_artists.append({"artist": h_line, "type": "h_wire"})

        label_obj = ax.text(
            x_wire_start - 0.1,
            ys[i],
            rf"$q_{{{i}}}$",
            ha="right",
            va="center",
            zorder=3,
        )
        text_artists.append({"artist": label_obj, "type": "wire_label"})

    # draw each gate
    for col_idx, col in enumerate(cols):
        x = xs[col_idx]
        for gate, qudits, dagger in col:
            if gate.color:
                fc = gate.color
            else:
                pauli_color, hs_color, cnot_color, cz_color, other_color = gate_colors
                nm = gate.name.upper().rstrip("†")
                if nm in {"I", "X", "Y", "Z"}:
                    fc = pauli_color
                elif nm in {"H", "S"}:
                    fc = hs_color
                elif nm.startswith("CNOT"):
                    fc = cnot_color
                elif nm.startswith("CZ"):
                    fc = cz_color
                elif nm == "M":
                    fc = "gray"
                else:
                    fc = other_color

            if gate.n_qudits == 1:
                y = ys[qudits[0]]

                if gate.name == "M":
                    box = patches.Rectangle(
                        (x - gate_w / 2, y - gate_h / 2),
                        gate_w,
                        gate_h,
                        facecolor=fc,
                        edgecolor=edgecolor,
                        zorder=2,
                    )
                    ax.add_patch(box)
                    line_artists.append({"artist": box, "type": "gate_edge"})
                    arc = patches.Arc(
                        (x, y - gate_h * 0.1),
                        width=gate_w * 0.7,
                        height=gate_h * 0.6,
                        angle=0,
                        theta1=0,
                        theta2=180,
                        color="white",
                        linewidth=3,
                        zorder=3,
                    )
                    ax.add_patch(arc)
                    line_artists.append({"artist": arc, "type": "meas_symbol"})

                    (arrow,) = ax.plot(
                        [x, x + gate_w * 0.35],
                        [y - gate_h * 0.1, y + gate_h * 0.25],
                        color="white",
                        linewidth=3,
                        zorder=3,
                    )
                    line_artists.append({"artist": arrow, "type": "meas_symbol"})
                elif gate.name == "NOISE_CHANNEL" or (
                    gate.a is not None and gate.b is not None
                ):
                    identity_condition = gate.a == 0 and gate.b == 0
                    base_radius = gate_w / 2
                    num_pts = 120
                    wiggle_amplitude = base_radius * 0.05
                    freq = 16
                    angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
                    radii = base_radius - wiggle_amplitude * np.cos(
                        freq * angles + np.pi
                    )
                    xs_edge = x + radii * np.cos(angles)
                    ys_edge = y + radii * np.sin(angles)

                    vertices = np.column_stack([xs_edge, ys_edge])
                    vertices = np.vstack([vertices, vertices[0]])
                    codes = (
                        [patches.Path.MOVETO]
                        + [patches.Path.LINETO] * (len(vertices) - 2)
                        + [patches.Path.CLOSEPOLY]
                    )
                    path = patches.Path(vertices, codes)
                    wiggly = patches.PathPatch(
                        path,
                        facecolor=fc if not identity_condition else "white",
                        edgecolor="black",
                        linewidth=1.5,
                        zorder=2,
                    )
                    ax.add_patch(wiggly)
                    line_artists.append({"artist": wiggly, "type": "noise_edge"})

                    if gate.a is not None and gate.b is not None:
                        if identity_condition:
                            txt = rf"$\mathit{{{'I'}}}$"
                        else:
                            if dagger:
                                txt = (
                                    rf"$\mathit{{W}}_{{{gate.a},{gate.b}}}^{{\dagger}}$"
                                )
                            else:
                                txt = rf"$\mathit{{W}}_{{{gate.a},{gate.b}}}$"
                        text_obj = ax.text(
                            x,
                            y,
                            txt,
                            ha="center",
                            va="center",
                            color=font_color,
                            zorder=3,
                        )
                        text_artists.append(
                            {
                                "artist": text_obj,
                                "type": "gate_label",
                                "name_len": (
                                    1 if identity_condition else len(gate.name) - 1
                                ),
                            }
                        )
                else:
                    box = patches.Rectangle(
                        (x - gate_w / 2, y - gate_h / 2),
                        gate_w,
                        gate_h,
                        facecolor=fc,
                        edgecolor=edgecolor,
                        zorder=2,
                    )
                    ax.add_patch(box)
                    line_artists.append({"artist": box, "type": "gate_edge"})
                    if dagger:
                        txt = rf"$\mathit{{{gate.name}}}^{{\dagger}}$"
                    else:
                        txt = rf"$\mathit{{{gate.name}}}$"
                    text_obj = ax.text(
                        x, y, txt, ha="center", va="center", color=font_color, zorder=3
                    )
                    text_artists.append(
                        {
                            "artist": text_obj,
                            "type": "gate_label",
                            "name_len": len(gate.name),
                        }
                    )
            else:
                if gate.name.startswith("CNOT"):
                    c, t = qudits
                    y_c, y_t = ys[c], ys[t]
                    (v_line,) = ax.plot(
                        [x, x],
                        [y_c, y_t],
                        linewidth=2,
                        color=fc,
                        zorder=1,
                        solid_capstyle="butt",
                    )
                    line_artists.append({"artist": v_line, "type": "v_wire"})

                    dot = patches.Circle(
                        (x, y_c), gate_h * 0.2, facecolor=fc, edgecolor=None, zorder=3
                    )
                    ax.add_patch(dot)

                    radius = gate_h * 0.3
                    target_circ = patches.Circle(
                        (x, y_t),
                        radius,
                        facecolor="white",
                        edgecolor=fc,
                        linewidth=2,
                        zorder=2,
                    )
                    ax.add_patch(target_circ)
                    line_artists.append(
                        {"artist": target_circ, "type": "cnot_symbol_line"}
                    )

                    (h_line_cross,) = ax.plot(
                        [x - radius, x + radius],
                        [y_t, y_t],
                        color=fc,
                        linewidth=2,
                        zorder=3,
                        solid_capstyle="butt",
                    )
                    line_artists.append(
                        {"artist": h_line_cross, "type": "cnot_symbol_line"}
                    )

                    if not dagger:
                        (v_line_cross,) = ax.plot(
                            [x, x],
                            [y_t - radius, y_t + radius],
                            color=fc,
                            linewidth=2,
                            zorder=3,
                            solid_capstyle="butt",
                        )
                        line_artists.append(
                            {"artist": v_line_cross, "type": "cnot_symbol_line"}
                        )
                elif gate.name.startswith("CZ"):
                    c, t = qudits
                    y_c, y_t = ys[c], ys[t]

                    # vertical line from control to the *edge* of the Z box on the target
                    # (the “edge” is whichever side is closer to the control wire)
                    sign = 1 if y_c > y_t else -1  # top wire has larger y
                    y_edge = y_t + sign * (gate_h * 0.5)

                    (v_line,) = ax.plot(
                        [x, x], [y_c, y_edge], linewidth=2, color=fc, zorder=1
                    )
                    line_artists.append({"artist": v_line, "type": "v_wire"})

                    # control dot (same style/size as CNOT's control)
                    dot = patches.Circle((x, y_c), gate_h * 0.2, facecolor=fc, zorder=3)
                    ax.add_patch(dot)

                    # target: Z box centered on the target wire
                    box = patches.Rectangle(
                        (x - gate_w / 2, y_t - gate_h / 2),
                        gate_w,
                        gate_h,
                        facecolor=fc,
                        zorder=2,
                    )
                    ax.add_patch(box)

                    if dagger:
                        txt = rf"$\mathit{{{'Z'}}}^{{\dagger}}$"
                    else:
                        txt = rf"$\mathit{{{'Z'}}}$"
                    text_obj = ax.text(
                        x,
                        y_t,
                        txt,
                        ha="center",
                        va="center",
                        color=font_color,
                        zorder=3,
                    )
                    text_artists.append(
                        {"artist": text_obj, "type": "gate_label", "name_len": 1}
                    )

                elif gate.name.startswith("SWAP"):
                    q1, q2 = qudits
                    y1, y2 = ys[q1], ys[q2]

                    # vertical connector
                    (v_line,) = ax.plot(
                        [x, x],
                        [y1, y2],
                        linewidth=2,
                        color="black",
                        zorder=1,
                        solid_capstyle="round",
                    )
                    line_artists.append({"artist": v_line, "type": "swap_v_wire"})

                    cross_scale = 0.3  # fraction of 1-qudit box; controls cross size
                    cross_half = cross_scale * min(gate_w, gate_h)

                    # upper cross
                    (l1u,) = ax.plot(
                        [x - cross_half, x + cross_half],
                        [y1 - cross_half, y1 + cross_half],
                        color="black",
                        linewidth=2,
                        zorder=3,
                        solid_capstyle="round",
                    )
                    (l2u,) = ax.plot(
                        [x - cross_half, x + cross_half],
                        [y1 + cross_half, y1 - cross_half],
                        color="black",
                        linewidth=2,
                        zorder=3,
                        solid_capstyle="round",
                    )
                    line_artists.append({"artist": l1u, "type": "swap_cross"})
                    line_artists.append({"artist": l2u, "type": "swap_cross"})

                    # lower cross
                    (l1d,) = ax.plot(
                        [x - cross_half, x + cross_half],
                        [y2 - cross_half, y2 + cross_half],
                        color="black",
                        linewidth=2,
                        zorder=3,
                        solid_capstyle="round",
                    )
                    (l2d,) = ax.plot(
                        [x - cross_half, x + cross_half],
                        [y2 + cross_half, y2 - cross_half],
                        color="black",
                        linewidth=2,
                        zorder=3,
                        solid_capstyle="round",
                    )
                    line_artists.append({"artist": l1d, "type": "swap_cross"})
                    line_artists.append({"artist": l2d, "type": "swap_cross"})
                elif gate.name.startswith("GROUP"):
                    gate_text = gate.name[6:]
                    if dagger:
                        gate_text = rf"$\mathit{{{gate_text}}}^{{\dagger}}$"
                    else:
                        gate_text = rf"$\mathit{{{gate_text}}}$"
                    start_q, end_q = min(qudits), max(qudits)
                    y_top = ys[start_q]
                    y_bottom = ys[end_q]

                    box_x = x - gate_w / 2
                    box_y = y_bottom - gate_h / 2
                    box_width = gate_w
                    box_height = y_top - y_bottom + gate_h

                    rect = patches.Rectangle(
                        (box_x, box_y),
                        box_width,
                        box_height,
                        facecolor=fc,
                        edgecolor=edgecolor,
                        zorder=1,
                    )
                    ax.add_patch(rect)
                    line_artists.append({"artist": rect, "type": "gate_edge"})

                    y_center = (y_top + y_bottom) / 2
                    text_obj = ax.text(
                        x,
                        y_center,
                        gate_text,
                        ha="center",
                        va="center",
                        rotation=90,
                        color="black",
                        zorder=3,
                    )
                    text_artists.append(
                        {
                            "artist": text_obj,
                            "type": "group_label",
                            "name_len": len(gate.name[6:]),
                            "n_qudits": qudits[1] - qudits[0] + 1,
                        }
                    )

    # finalize plot
    ax.set_xlim(0, total_width)
    ax.set_ylim(-base_spacing, qc.n * base_spacing)
    ax.axis("off")

    # --- DYNAMIC FONT SIZING HANDLER ---
    def on_resize(event):
        ax_bbox = ax.get_window_extent()
        if ax_bbox.width == 0 or ax_bbox.height == 0:
            return

        y_data_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        y_scale = ax_bbox.height / y_data_range
        dpi = fig.get_dpi()

        # line widths
        # -------------------------------------------------------------------
        gate_h_pixels = gate_h * y_scale
        h_wire_lw = max(0.5, gate_h_pixels * 0.03)  # 0.xx% of gate height
        v_wire_lw = max(1.0, gate_h_pixels * 0.06)
        meas_lw = max(1.0, gate_h_pixels * 0.03)

        noise_edge_lw = max(0.5, gate_h_pixels * 0.03)

        for item in line_artists:
            artist = item["artist"]
            if item["type"] == "h_wire":
                artist.set_linewidth(h_wire_lw)
            elif item["type"] == "v_wire":
                artist.set_linewidth(v_wire_lw)
            elif item["type"] == "meas_symbol":
                artist.set_linewidth(meas_lw)
            elif item["type"] == "cnot_symbol_line":
                artist.set_linewidth(v_wire_lw)
            elif item["type"] == "noise_edge":
                artist.set_linewidth(noise_edge_lw)
            elif item["type"] == "gate_edge":
                artist.set_linewidth(h_wire_lw)
            elif item["type"] == "swap_v_wire":
                artist.set_linewidth(v_wire_lw)
            elif item["type"] == "swap_cross":
                artist.set_linewidth(v_wire_lw)
        # -------------------------------------------------------------------
        for item in text_artists:
            artist = item["artist"]

            if item["type"] == "title":
                font_size_pt = fig.get_figwidth() * 1.5
                artist.set_fontsize(font_size_pt)

            elif item["type"] == "wire_label":
                wire_lane_h_pixels = base_spacing * y_scale
                font_height_pixels = wire_lane_h_pixels * 0.3
                font_size_pt = font_height_pixels * (72 / dpi)
                artist.set_fontsize(font_size_pt)

            elif item["type"] == "gate_label":
                gate_h_pixels = gate_h * y_scale
                font_height_pixels = (gate_h_pixels * 0.6) / np.sqrt(item["name_len"])
                font_size_pt = font_height_pixels * (72 / dpi)
                artist.set_fontsize(font_size_pt)

            elif item["type"] == "cnot_symbol":
                circ_diameter_pixels = (gate_h * 0.3 * 2) * y_scale
                font_height_pixels = circ_diameter_pixels * 0.9
                font_size_pt = font_height_pixels * (72 / dpi)
                artist.set_fontsize(font_size_pt)

            elif item["type"] == "group_label":
                gate_h_pixels = gate_h * y_scale
                font_height_pixels = (
                    gate_h_pixels * 0.6
                )  # optionally normalize by np.sqrt(np.sqrt(item['name_len']))
                font_size_pt = font_height_pixels * (72 / dpi)
                artist.set_fontsize(font_size_pt)

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("resize_event", on_resize)
    ax.callbacks.connect("xlim_changed", on_resize)
    ax.callbacks.connect("ylim_changed", on_resize)

    plt.tight_layout()

    fig.canvas.draw()

    on_resize(None)

    if save_path:
        fig.tight_layout()
        fig.canvas.draw()
        on_resize(None)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        print(f"\nSaved circuit diagram to '{save_path}'")

    if show_image:
        if show_info or show_title:
            fig.subplots_adjust(top=0.92)
            fig.canvas.draw()
        on_resize(None)
        plt.show()
