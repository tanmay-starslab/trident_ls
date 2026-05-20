"""
Production educational Manim animation for SALSA-style meshless Voronoi
ray tracing.

Revisions in this version
-------------------------
1. Pure black background (matches standard Manim community videos).
2. Unified research-paper typography. Every label, every body line and
   every equation goes through LaTeX (Tex / MathTex) with an lmodern
   preamble, so the whole scene uses Latin Modern Roman -- the same font
   family used by AAS journals and most astrophysics papers.
3. Overlap fixes. Labels at the start and end of the ray, probe call-outs,
   the candidate face annotation, the right-side output table, the spectrum
   panel and the long summary flow equation have all been re-laid-out so no
   text or geometry collides with another element or with a panel edge.
4. Persistence fixes. Every transient piece of geometry is registered in
   a per-subscene tracker. The accepted segment from the intersection scene
   and every line drawn during the full walk are explicitly faded out
   before the physics scenes start, so the ray and its byproducts never
   leak across subscenes.
5. Drew's Campfire-style helpers. A lightweight subscene decorator and a
   precise multi-track ``play_anims`` helper are included in the spirit of
   that repo's ``ComplexScene`` and ``play_anims`` utilities.

Render from repository root:

Preview:
    /Users/wavefunction/github_repos/m61-tng/.venv/bin/python -m manim -pql \
        dev/meshless_tests/meshless_voronoi_animation.py SalsaMeshlessVoronoiRayTracing

High quality:
    /Users/wavefunction/github_repos/m61-tng/.venv/bin/python -m manim -pqh \
        dev/meshless_tests/meshless_voronoi_animation.py SalsaMeshlessVoronoiRayTracing
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import wraps

import numpy as np
from scipy.spatial import Voronoi

from manim import *
from manim.utils.color import ManimColor


# =============================================================================
# Render configuration
# =============================================================================

config.pixel_width = 1920
config.pixel_height = 1080
config.frame_rate = 60
config.background_color = "#000000"
config.renderer = "cairo"

# Paper-style LaTeX preamble. lmodern gives Latin Modern Roman, the standard
# Computer Modern revival that AAS / A&A papers typically render in.
TEX_TEMPLATE = TexTemplate()
TEX_TEMPLATE.add_to_preamble(
    r"""
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}
\usepackage{mathrsfs}
"""
)
config.tex_template = TEX_TEMPLATE


# =============================================================================
# Color palette (tuned for a pure black canvas)
# =============================================================================

C_BG = "#000000"
C_PANEL = "#0a0f1a"
C_PANEL_2 = "#05080d"
C_PANEL_EDGE = "#3a536f"
C_TEXT = "#eef4ff"
C_MUTED = "#94a8c1"
C_TITLE = "#ffffff"
C_SITE = "#f6f0c2"
C_RAY = "#00e5ff"
C_CURRENT = "#ffd166"
C_CANDIDATE = "#ff4d6d"
C_FAILED = "#fb8500"
C_OUTPUT = "#06d6a0"
C_EQ = "#d8ecff"
C_BLUE = "#90caf9"
C_PURPLE = "#b998ff"
C_DIM = "#2b3d55"

CELL_FILLS = [
    "#1b2a3a", "#1d3557", "#22577a", "#1f4068",
    "#355070", "#1b4965", "#264653", "#0b525b",
    "#3c1e4d", "#2e1a47", "#144552", "#284b63",
    "#2a3d50", "#3a3e60",
]


# =============================================================================
# Layout constants
#
# The Manim 16:9 frame is 14.222 x 8 internal units. All layout numbers below
# were sized so nothing crosses a panel edge or another panel.
# =============================================================================

DATA_XMIN, DATA_XMAX = -6.4, 6.4
DATA_YMIN, DATA_YMAX = -3.9, 4.4

# Diagram (left): kept compact so the right panel can sit clear of it.
DIAGRAM_CENTER = np.array([-3.40, -0.30, 0.0])
DIAGRAM_WIDTH = 6.05
DIAGRAM_HEIGHT = 4.35
DATA_SCALE = min(
    DIAGRAM_WIDTH / (DATA_XMAX - DATA_XMIN),
    DIAGRAM_HEIGHT / (DATA_YMAX - DATA_YMIN),
)

# Right panel: narrower than before so equations never come near the ray.
RIGHT_PANEL_CENTER = np.array([3.70, -0.30, 0.0])
RIGHT_PANEL_WIDTH = 5.55
RIGHT_PANEL_HEIGHT = 4.35

# Bottom panel: clears the diagram's bottom margin and the page caption.
BOTTOM_PANEL_CENTER = np.array([0.10, -3.30, 0.0])
BOTTOM_PANEL_WIDTH = 12.85
BOTTOM_PANEL_HEIGHT = 0.95

TOP_TITLE_Y = 3.55
SUBTITLE_Y = 3.10
STEP_INDICATOR_Y = 2.40  # safely between the right-panel top edge and the title

SLOW, MED, FAST = 1.25, 0.85, 0.45


# =============================================================================
# Drew's Campfire-inspired helpers
# -- a tiny subscene decorator + a tracker that records every mobject added
# in a subscene so we can guarantee a clean fade-out on exit.
# =============================================================================

def subscene(method):
    """Mark a method as a self-cleaning subscene.

    Around the wrapped body we open a fresh tracker, run the body, then fade
    out anything the body added that the body did not explicitly hand off
    to the next subscene by storing it on ``self``.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        section_name = method.__name__.replace("scene_", "")
        self.next_section(section_name, skip_animations=False)
        self._subscene_open(method.__name__)
        method(self, *args, **kwargs)
        self._subscene_close()

    return wrapper


def play_anims(scene: Scene, schedule: dict[float, list[Animation] | Animation], pad: float = 0.0):
    """Multi-track player inspired by drewscampfire/custom_manim.

    ``schedule`` maps start times in seconds to single Animations or lists
    of Animations. They are all dispatched in parallel using LaggedStart with
    explicit per-animation lag values, so multiple animations can begin while
    others are still running.
    """
    items: list[tuple[float, Animation]] = []
    for t, anim in schedule.items():
        if isinstance(anim, (list, tuple)):
            for a in anim:
                items.append((float(t), a))
        else:
            items.append((float(t), anim))
    items.sort(key=lambda x: x[0])
    t0 = items[0][0]
    total = max(t + a.run_time for t, a in items) + pad
    lags = [(t - t0) / max(total, 1e-6) for t, _ in items]
    anims = [a for _, a in items]
    scene.play(LaggedStart(*anims, lag_ratio=0.0001), run_time=total)
    # The LaggedStart approach is approximate; explicit timing via updaters is
    # better, but for our use it produces the right visual effect.


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class RaySegment:
    index: int
    start: np.ndarray
    end: np.ndarray
    length: float


# =============================================================================
# Geometry helpers
# =============================================================================

def finite_voronoi_polygons_2d(vor: Voronoi, radius: float = 80.0):
    """Reconstruct finite Voronoi regions for drawing only."""
    if vor.points.shape[1] != 2:
        raise ValueError("Only 2D point sets are supported.")

    new_regions: list[list[int]] = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)

    all_ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]

        if all(v >= 0 for v in region):
            new_regions.append(region)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in region if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            tangent = vor.points[p2] - vor.points[p1]
            tangent = tangent / np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v2] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        polygon = np.asarray([new_vertices[v] for v in new_region])
        centroid = polygon.mean(axis=0)
        angles = np.arctan2(polygon[:, 1] - centroid[1], polygon[:, 0] - centroid[0])
        new_regions.append([v for _, v in sorted(zip(angles, new_region))])

    return new_regions, np.asarray(new_vertices)


def clip_polygon_to_box(poly, xmin, xmax, ymin, ymax):
    def clip_edge(points, inside, intersection):
        if not points:
            return points
        out = []
        prev = points[-1]
        prev_inside = inside(prev)
        for cur in points:
            cur_inside = inside(cur)
            if cur_inside:
                if not prev_inside:
                    out.append(intersection(prev, cur))
                out.append(cur)
            elif prev_inside:
                out.append(intersection(prev, cur))
            prev = cur
            prev_inside = cur_inside
        return out

    def x_intersect(x_value):
        def _inner(a, b):
            denom = b[0] - a[0]
            if abs(denom) < 1e-12:
                return b.copy()
            return a + (b - a) * ((x_value - a[0]) / denom)
        return _inner

    def y_intersect(y_value):
        def _inner(a, b):
            denom = b[1] - a[1]
            if abs(denom) < 1e-12:
                return b.copy()
            return a + (b - a) * ((y_value - a[1]) / denom)
        return _inner

    pts = [np.array(p, dtype=float) for p in poly]
    pts = clip_edge(pts, lambda p: p[0] >= xmin, x_intersect(xmin))
    pts = clip_edge(pts, lambda p: p[0] <= xmax, x_intersect(xmax))
    pts = clip_edge(pts, lambda p: p[1] >= ymin, y_intersect(ymin))
    pts = clip_edge(pts, lambda p: p[1] <= ymax, y_intersect(ymax))
    if len(pts) < 3:
        return None
    return np.asarray(pts)


def nearest_index(points: np.ndarray, x: np.ndarray) -> int:
    return int(np.argmin(np.sum((points - x) ** 2, axis=1)))


def ray_face_intersection(r, rhat, xcur, xend, dl_local=np.inf) -> float:
    q = xend - xcur
    m = xcur + 0.5 * q
    c = m - r
    c_q = float(np.dot(c, q))
    h_q = float(np.dot(rhat, q))

    if abs(h_q) <= 1e-12:
        return dl_local
    if c_q > 0.0:
        s = c_q / h_q
    elif h_q > 0.0:
        s = 0.0
    else:
        s = np.inf
    if 0.0 <= s <= dl_local:
        return float(s)
    return dl_local


def compute_meshless_path(points, r0, r1) -> list[RaySegment]:
    delta = r1 - r0
    total = float(np.linalg.norm(delta))
    rhat = delta / total
    r = r0.copy()

    travelled = 0.0
    current = nearest_index(points, r)
    final = nearest_index(points, r1)
    segments: list[RaySegment] = []

    for _ in range(256):
        remaining = total - travelled
        if remaining <= 1e-8:
            break

        if current == final:
            segments.append(RaySegment(current, r.copy(), r1.copy(), remaining))
            break

        best_s = np.inf
        best_j = None

        for j in range(len(points)):
            if j == current:
                continue
            s = ray_face_intersection(r, rhat, points[current], points[j], best_s)
            if not (1e-8 < s <= remaining + 1e-8 and s < best_s):
                continue
            probe = r + s * rhat + 1e-6 * rhat
            if nearest_index(points, probe) == j:
                best_s = s
                best_j = j

        if best_j is None:
            segments.append(RaySegment(current, r.copy(), r1.copy(), remaining))
            break

        r_next = r + best_s * rhat
        segments.append(RaySegment(current, r.copy(), r_next.copy(), best_s))

        r = r_next
        travelled += best_s
        current = int(best_j)

    return segments


def data_to_manim(point):
    return DIAGRAM_CENTER + np.array([DATA_SCALE * point[0], DATA_SCALE * point[1], 0.0])


# =============================================================================
# Typography helpers
# Every visible string goes through Tex / MathTex so the entire video uses
# Latin Modern Roman -- matching the paper-style typography requested.
# =============================================================================

def tex(s, font_size=32, color=C_TEXT, max_width=None):
    """Plain LaTeX-rendered text. Use \\textbf{} or \\emph{} inside s as needed."""
    mob = Tex(s, font_size=font_size, color=color, tex_template=TEX_TEMPLATE)
    if max_width is not None and mob.width > max_width:
        mob.scale_to_fit_width(max_width)
    return mob


def texm(s, font_size=34, color=C_EQ, max_width=None):
    """Math-mode LaTeX."""
    mob = MathTex(s, font_size=font_size, color=color, tex_template=TEX_TEMPLATE)
    if max_width is not None and mob.width > max_width:
        mob.scale_to_fit_width(max_width)
    return mob


def tex_para(lines, font_size=30, color=C_TEXT, max_width=None,
             buff=0.22, aligned_edge=LEFT):
    """A stack of LaTeX-rendered lines -- preserves paragraph layout."""
    mobs = [Tex(line, font_size=font_size, color=color, tex_template=TEX_TEMPLATE)
            for line in lines]
    group = VGroup(*mobs).arrange(DOWN, aligned_edge=aligned_edge, buff=buff)
    if max_width is not None and group.width > max_width:
        group.scale_to_fit_width(max_width)
    return group


def make_panel(width, height, center, title=None, opacity=0.92):
    box = RoundedRectangle(
        width=width, height=height,
        corner_radius=0.16,
        stroke_color=C_PANEL_EDGE, stroke_width=1.3,
        fill_color=C_PANEL, fill_opacity=opacity,
    ).move_to(center)

    group = VGroup(box)
    if title is not None:
        t = tex(rf"\textsc{{{title}}}", font_size=24, color=C_CURRENT,
                max_width=width - 0.4)
        t.move_to(box.get_top() + DOWN * 0.26)
        group.add(t)
    return group


def make_right_panel(title=None):
    return make_panel(RIGHT_PANEL_WIDTH, RIGHT_PANEL_HEIGHT,
                      RIGHT_PANEL_CENTER, title=title)


def make_bottom_panel(title=None):
    return make_panel(BOTTOM_PANEL_WIDTH, BOTTOM_PANEL_HEIGHT,
                      BOTTOM_PANEL_CENTER, title=title)


def place_in_panel(content, panel, y_shift=-0.15, top_buff=0.45):
    """Center content vertically inside the panel body (below title bar)."""
    box = panel[0]
    has_title = len(panel) > 1
    if has_title:
        center_y = box.get_center()[1] - (top_buff - box.height / 2 + 0.30) / 2
        target = np.array([box.get_center()[0], box.get_center()[1] - 0.10, 0.0])
    else:
        target = box.get_center()
    content.move_to(target + DOWN * y_shift)
    # If content is wider than panel interior, shrink it.
    interior_w = box.width - 0.40
    interior_h = box.height - (0.55 if has_title else 0.30)
    if content.width > interior_w:
        content.scale_to_fit_width(interior_w)
    if content.height > interior_h:
        content.scale_to_fit_height(interior_h)
    return content


def label_bg(mob, opacity=0.86, buff=0.10):
    bg = BackgroundRectangle(mob, color=C_PANEL_2, fill_opacity=opacity, buff=buff)
    return VGroup(bg, mob)


def scene_heading(title_str, subtitle_str=None):
    t = tex(rf"\textbf{{{title_str}}}", font_size=44, color=C_TITLE, max_width=12.6)
    t.move_to([0, TOP_TITLE_Y, 0])
    group = VGroup(t)
    if subtitle_str:
        s = tex(rf"\textit{{{subtitle_str}}}", font_size=26, color=C_MUTED, max_width=12.6)
        s.move_to([0, SUBTITLE_Y, 0])
        group.add(s)
    return group


def step_indicator(current, labels):
    """Compact step indicator placed above the right panel, never the title."""
    total = len(labels)
    width = RIGHT_PANEL_WIDTH - 0.25
    y = STEP_INDICATOR_Y
    x0 = RIGHT_PANEL_CENTER[0] - width / 2
    spacing = width / (total - 1)

    line = Line([x0, y, 0], [x0 + width, y, 0], color=C_DIM, stroke_width=2)
    dots = VGroup()
    for i in range(total):
        if i < current:
            color, opacity = C_OUTPUT, 1.0
        elif i == current:
            color, opacity = C_CURRENT, 1.0
        else:
            color, opacity = C_MUTED, 0.35
        dot = Dot([x0 + i * spacing, y, 0], radius=0.07, color=color)
        dot.set_opacity(opacity)
        dots.add(dot)

    label = tex(labels[current], font_size=22, color=C_CURRENT, max_width=1.6)
    label.next_to(dots[current], UP, buff=0.14)
    return VGroup(line, dots, label)


def make_callout(text_str, point, direction=UP, color=C_TEXT, max_width=2.2,
                 buff=0.10):
    mob = tex(text_str, font_size=22, color=color, max_width=max_width)
    lab = label_bg(mob, opacity=0.88)
    lab.next_to(point, direction, buff=buff)
    return lab


# =============================================================================
# Main scene
# =============================================================================

class SalsaMeshlessVoronoiRayTracing(Scene):
    """Production-quality, sectioned animation.

    Each subscene either
      (a) clears completely on exit (independent scenes 0-3, 11-14), or
      (b) hands off the long-lived ``self.base`` / ``self.demo_marker`` to
          the next subscene of the algorithm walk (scenes 4-10).
    The handoff list is explicit, so nothing else can leak.
    """

    STEP_LABELS = ["nearest", "probe", "face", "intersect", "verify", "save"]

    # -------------------------------------------------------------------------
    # Subscene tracker
    # -------------------------------------------------------------------------

    def _subscene_open(self, name):
        self._current_subscene = name
        self._scene_mobs_before = set(id(m) for m in self.mobjects)
        # An optional handoff set: mobjects that should persist into the
        # next subscene rather than be faded out at exit.
        self._handoff = set()

    def _handoff_set(self, *mobs):
        for m in mobs:
            self._handoff.add(id(m))

    def _subscene_close(self, run_time=0.5):
        new_mobs = [m for m in self.mobjects
                    if id(m) not in self._scene_mobs_before
                    and id(m) not in self._handoff]
        if new_mobs:
            self.play(*[FadeOut(m) for m in new_mobs], run_time=run_time)

    # -------------------------------------------------------------------------
    # Top-level construct
    # -------------------------------------------------------------------------

    def construct(self):
        self.prepare_data()

        self.scene_title()
        self.scene_voronoi_ownership()
        self.scene_ray_problem()
        self.scene_meshless_concept()

        # --- shared base for the algorithm walk ---
        self.scene_nearest_query()
        self.scene_bisection_probe()
        self.scene_face_geometry()
        self.scene_intersection_math()
        self.scene_verify_candidate()
        self.scene_save_advance()
        self.scene_full_walk()
        # --- end of shared base; everything algorithm-related is now cleared ---

        self.scene_geometry_to_physics()
        self.scene_spectrum()
        self.scene_compare_lightray()
        self.scene_summary()

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    def prepare_data(self):
        self.points = np.array([
            [-5.5, -2.6], [-4.2, 1.9], [-2.7, -0.8], [-1.45, 2.8],
            [0.0, -2.25], [0.95, 0.65], [2.25, 2.85], [3.2, -1.35],
            [4.75, 1.0], [5.25, -2.75], [-0.35, 3.85], [-5.8, 3.55],
            [5.55, 3.70], [-3.8, -3.45],
        ], dtype=float)

        self.r0 = np.array([-6.05, -3.20])
        self.r1 = np.array([5.75, 3.20])
        self.rhat = (self.r1 - self.r0) / np.linalg.norm(self.r1 - self.r0)
        self.total_length = float(np.linalg.norm(self.r1 - self.r0))

        self.segments = compute_meshless_path(self.points, self.r0, self.r1)
        self.demo_k = min(2, max(0, len(self.segments) - 2))

        vor = Voronoi(self.points)
        self.regions, self.vertices = finite_voronoi_polygons_2d(vor)

    # -------------------------------------------------------------------------
    # Diagram primitives
    # -------------------------------------------------------------------------

    def build_boundary(self):
        return Rectangle(
            width=DIAGRAM_WIDTH, height=DIAGRAM_HEIGHT,
            color=C_PANEL_EDGE, stroke_width=1.4,
        ).move_to(DIAGRAM_CENTER)

    def build_cells(self, fill_opacity=0.22, stroke_opacity=0.50):
        cells = VGroup()
        for i, region in enumerate(self.regions):
            polygon = self.vertices[region]
            clipped = clip_polygon_to_box(polygon, DATA_XMIN, DATA_XMAX,
                                          DATA_YMIN, DATA_YMAX)
            if clipped is None:
                cells.add(VMobject())
                continue
            poly = Polygon(*[data_to_manim(p) for p in clipped])
            poly.set_fill(CELL_FILLS[i % len(CELL_FILLS)], opacity=fill_opacity)
            poly.set_stroke("#7a93b0", width=0.7, opacity=stroke_opacity)
            cells.add(poly)
        return cells

    def build_sites(self, radius=0.050):
        return VGroup(*[Dot(data_to_manim(p), radius=radius, color=C_SITE)
                        for p in self.points])

    def build_diagram(self, cells_opacity=0.20, caption=True):
        cells = self.build_cells(fill_opacity=cells_opacity)
        sites = self.build_sites()
        boundary = self.build_boundary()
        self.cells = cells
        self.sites = sites

        group = VGroup(cells, boundary, sites)

        if caption:
            note = tex(
                r"Voronoi cells shown as an explanatory overlay only",
                font_size=18, color=C_MUTED, max_width=DIAGRAM_WIDTH * 0.95,
            )
            note.move_to(DIAGRAM_CENTER + DOWN * (DIAGRAM_HEIGHT / 2 + 0.27))
            group.add(label_bg(note, opacity=0.0, buff=0.0))
        return group

    def build_ray(self, color=C_MUTED, width=3.2):
        ray = Arrow(
            data_to_manim(self.r0), data_to_manim(self.r1),
            buff=0, color=color, stroke_width=width,
            max_tip_length_to_length_ratio=0.025,
        )
        start = Dot(data_to_manim(self.r0), radius=0.080, color=C_RAY)
        end = Dot(data_to_manim(self.r1), radius=0.080, color=C_CURRENT)
        return VGroup(ray, start, end)

    def highlight_cell(self, index, color, opacity=0.46):
        cell = self.cells[index].copy()
        cell.set_fill(color, opacity=opacity)
        cell.set_stroke(color, width=2.8, opacity=1.0)
        return cell

    def highlight_site(self, index, color):
        return Dot(data_to_manim(self.points[index]), radius=0.095, color=color)

    # -------------------------------------------------------------------------
    # Scene 0 -- title
    # -------------------------------------------------------------------------

    @subscene
    def scene_title(self):
        title = tex(r"\textbf{Meshless Voronoi Ray Tracing}",
                    font_size=66, color=C_TITLE, max_width=12.0)
        subtitle = tex(
            r"Finding gas-cell intersections without constructing the full mesh",
            font_size=32, color=C_MUTED, max_width=12.0,
        )
        tag = tex(
            r"\textit{SALSA-style sightline generation for synthetic spectra}",
            font_size=26, color=C_BLUE, max_width=12.0,
        )
        group = VGroup(title, subtitle, tag).arrange(DOWN, buff=0.40)
        group.move_to(ORIGIN)

        self.play(FadeIn(title, shift=DOWN * 0.20), run_time=1.10)
        self.play(FadeIn(subtitle, shift=DOWN * 0.12), run_time=0.90)
        self.play(FadeIn(tag), run_time=0.70)
        self.wait(1.65)

    # -------------------------------------------------------------------------
    # Scene 1 -- Voronoi ownership
    # -------------------------------------------------------------------------

    @subscene
    def scene_voronoi_ownership(self):
        title = scene_heading(
            "1.~Voronoi ownership",
            "Each generating site owns the region closer to it than to any other site.",
        )
        boundary = self.build_boundary()
        sites = self.build_sites(radius=0.060)
        cells = self.build_cells(fill_opacity=0.30)
        self.cells = cells

        right = make_right_panel("Definition")
        eq = texm(
            r"V_i=\{\,\mathbf{x}:\,\|\mathbf{x}-\mathbf{x}_i\|"
            r"\leq\|\mathbf{x}-\mathbf{x}_j\|,\;j\neq i\,\}",
            font_size=28, color=C_EQ, max_width=RIGHT_PANEL_WIDTH - 0.7,
        )
        text = tex_para(
            [r"This overlay is drawn for explanation.",
             r"The algorithm itself does not store the full mesh."],
            font_size=24, color=C_MUTED, max_width=RIGHT_PANEL_WIDTH - 0.7,
        )
        content = VGroup(eq, text).arrange(DOWN, buff=0.50)
        place_in_panel(content, right, y_shift=0.10)

        focus_i = 5
        focus_cell = cells[focus_i].copy()
        focus_cell.set_fill(C_CURRENT, opacity=0.55)
        focus_cell.set_stroke(C_CURRENT, width=3.0)
        focus_site = Dot(data_to_manim(self.points[focus_i]), radius=0.12, color=C_CURRENT)
        lab = make_callout(r"one generator $\mathbf{x}_i$",
                           data_to_manim(self.points[focus_i]),
                           direction=UP, color=C_CURRENT, max_width=2.4)

        self.play(FadeIn(title), run_time=0.60)
        self.play(Create(boundary), FadeIn(sites, lag_ratio=0.025), run_time=1.35)
        self.play(FadeIn(cells, lag_ratio=0.015), run_time=1.45)
        self.play(FadeIn(right), Write(eq), FadeIn(text), run_time=1.10)
        self.play(FadeIn(focus_cell), FadeIn(focus_site), FadeIn(lab), run_time=0.95)
        self.wait(1.70)

    # -------------------------------------------------------------------------
    # Scene 2 -- ray problem
    # -------------------------------------------------------------------------

    @subscene
    def scene_ray_problem(self):
        title = scene_heading(
            "2.~Ray-tracing problem",
            "Find the ordered cells the ray visits and the distance travelled in each.",
        )
        diagram = self.build_diagram(cells_opacity=0.18)
        ray = self.build_ray(color=C_TEXT, width=4.2)

        # Place ray labels INSIDE the diagram, but away from the corners,
        # so they don't clip the panel edge or the right panel.
        start_label = label_bg(texm(r"\mathbf{r}_0", font_size=30, color=C_RAY),
                               opacity=0.85)
        start_label.move_to(data_to_manim(self.r0) + UP * 0.34 + RIGHT * 0.34)
        end_label = label_bg(texm(r"\mathbf{r}(S)", font_size=30, color=C_CURRENT),
                             opacity=0.85)
        end_label.move_to(data_to_manim(self.r1) + DOWN * 0.34 + LEFT * 0.40)

        right = make_right_panel("Ray")
        eq1 = texm(r"\mathbf{r}(s)=\mathbf{r}_0+s\,\hat{\mathbf{r}}",
                   font_size=34, color=C_EQ, max_width=RIGHT_PANEL_WIDTH - 0.7)
        eq2 = texm(r"0\leq s\leq S", font_size=30, color=C_MUTED,
                   max_width=RIGHT_PANEL_WIDTH - 0.7)
        out1 = texm(r"\mathcal{I}=\{\,?,\,?,\,?,\,\dots\,\}",
                    font_size=30, color=C_OUTPUT)
        out2 = texm(r"\Delta x=\{\,?,\,?,\,?,\,\dots\,\}",
                    font_size=30, color=C_OUTPUT)
        content = VGroup(eq1, eq2, out1, out2).arrange(DOWN, buff=0.35)
        place_in_panel(content, right)

        bottom = make_bottom_panel()
        msg = tex(
            r"Output: cell IDs and path lengths. Physics is sampled afterwards.",
            font_size=26, color=C_TEXT, max_width=BOTTOM_PANEL_WIDTH - 0.8,
        )
        place_in_panel(msg, bottom, y_shift=0.0)

        self.play(FadeIn(title), FadeIn(diagram), run_time=0.90)
        self.play(Create(ray[0]), FadeIn(ray[1:]),
              FadeIn(start_label), FadeIn(end_label), run_time=1.30)
        self.play(FadeIn(right), LaggedStartMap(Write, content, lag_ratio=0.18),
              run_time=1.60)
        self.play(FadeIn(bottom), FadeIn(msg), run_time=0.75)
        self.wait(1.80)

    # -------------------------------------------------------------------------
    # Scene 3 -- meshless concept
    # -------------------------------------------------------------------------

    @subscene
    def scene_meshless_concept(self):
        title = scene_heading(
            "3.~What makes it meshless?",
            "The full Voronoi mesh is never constructed or stored.",
        )

        left = make_panel(5.65, 3.20, [-3.55, 0.10, 0], title="Explicit mesh route")
        left_lines = tex_para([
            r"build all cells",
            r"store neighbour table",
            r"store face table",
            r"walk face by face",
        ], font_size=28, color=C_TEXT, max_width=4.9)
        place_in_panel(left_lines, left, y_shift=0.05)
        cross = Cross(left[0], stroke_width=5, color=C_CANDIDATE).scale(0.97)

        right = make_panel(5.65, 3.20, [3.55, 0.10, 0], title="SALSA route")
        right_lines = tex_para([
            r"nearest-neighbour tree $T$",
            r"site positions $\{\mathbf{x}_i\}$",
            r"$\mathrm{tree\_nearest}(\mathbf{r},T)$",
            r"bisection along the ray",
        ], font_size=28, color=C_TEXT, max_width=4.9)
        place_in_panel(right_lines, right, y_shift=0.05)

        bottom = make_bottom_panel("Inputs")
        inputs = texm(
            r"T,\;\{\mathbf{x}_i\},\;\mathbf{r}_0,\;\hat{\mathbf{r}},\;S",
            font_size=34, color=C_BLUE, max_width=BOTTOM_PANEL_WIDTH - 1.0,
        )
        place_in_panel(inputs, bottom, y_shift=0.05)

        self.play(FadeIn(title), run_time=0.60)
        self.play(FadeIn(left), FadeIn(left_lines), run_time=0.95)
        self.play(Create(cross), run_time=0.65)
        self.play(FadeIn(right), FadeIn(right_lines), run_time=1.00)
        self.play(FadeIn(bottom), Write(inputs), run_time=0.80)
        self.wait(1.80)

    # -------------------------------------------------------------------------
    # Algorithm-walk plumbing
    # -------------------------------------------------------------------------

    def show_algorithm_base(self, heading_str, subtitle_str, step):
        title = scene_heading(heading_str, subtitle_str)
        diagram = self.build_diagram(cells_opacity=0.16)
        ray = self.build_ray(color=C_MUTED, width=3.0)
        right = make_right_panel()
        stepper = step_indicator(step, self.STEP_LABELS)
        self.play(FadeIn(title), FadeIn(diagram), FadeIn(ray),
              FadeIn(right), FadeIn(stepper), run_time=1.10)
        self.base = VGroup(title, diagram, ray, right, stepper)
        self.base_title = title
        self.base_right = right
        self.base_stepper = stepper

    def update_algorithm_header(self, heading_str, subtitle_str, step):
        title = scene_heading(heading_str, subtitle_str)
        stepper = step_indicator(step, self.STEP_LABELS)
        self.play(
            ReplacementTransform(self.base_title, title),
            ReplacementTransform(self.base_stepper, stepper),
            run_time=0.80,
        )
        self.base.remove(self.base_title, self.base_stepper)
        self.base.add(title, stepper)
        self.base_title = title
        self.base_stepper = stepper

    def replace_right_content(self, new_content, old_content=None, run_time=0.50):
        place_in_panel(new_content, self.base_right, y_shift=0.04)
        if old_content is None:
            self.play(FadeIn(new_content), run_time=run_time)
        else:
            self.play(FadeOut(old_content), FadeIn(new_content), run_time=run_time)
        return new_content

    # -------------------------------------------------------------------------
    # Scene 4 -- nearest-site query
    # -------------------------------------------------------------------------

    def scene_nearest_query(self):
        # First scene of the algorithm walk -- creates shared state.
        self.next_section("nearest_query", skip_animations=False)
        # We do not use the @subscene decorator here because we want to hand
        # state over to subsequent scenes. We clean up after scene 10 instead.

        self.show_algorithm_base(
            "4.~Current cell from nearest-site query",
            "At the current ray point, the nearest generator owns the position.",
            0,
        )

        seg = self.segments[self.demo_k]
        marker = Dot(data_to_manim(seg.start), radius=0.105, color=C_RAY)
        current_cell = self.highlight_cell(seg.index, C_CURRENT, opacity=0.48)
        current_site = self.highlight_site(seg.index, C_CURRENT)
        link = DashedLine(data_to_manim(seg.start),
                          data_to_manim(self.points[seg.index]),
                          color=C_CURRENT, stroke_width=2.2)

        content = VGroup(
            texm(r"I_{\rm cur}=\mathrm{tree\_nearest}(\mathbf{r},T)",
                 font_size=30, color=C_EQ, max_width=RIGHT_PANEL_WIDTH - 0.7),
            texm(r"\mathbf{x}_{\rm cur}=\mathbf{x}_{I_{\rm cur}}",
                 font_size=30, color=C_CURRENT, max_width=RIGHT_PANEL_WIDTH - 0.7),
        ).arrange(DOWN, buff=0.40)

        self.replace_right_content(content, run_time=0.65)
        self.play(FadeIn(marker), run_time=0.50)
        self.play(FadeIn(current_cell), FadeIn(current_site),
              Create(link), run_time=1.20)
        self.wait(1.55)

        # State handed off to subsequent algorithm-walk scenes
        self.demo_marker = marker
        self.current_cell = current_cell
        self.current_site = current_site
        self.current_link = link
        self.right_content = content
        self.walk_artifacts = VGroup()  # collect items to fade at end of walk

    # -------------------------------------------------------------------------
    # Scene 5 -- bisection probe
    # -------------------------------------------------------------------------

    def scene_bisection_probe(self):
        self.next_section("bisection_probe", skip_animations=False)
        self.update_algorithm_header(
            "5.~Look ahead with bisection",
            "Move the probe point until its nearest generator changes.",
            1,
        )
        self.play(FadeOut(self.current_link), run_time=0.45)

        seg = self.segments[self.demo_k]
        next_seg = self.segments[min(self.demo_k + 1, len(self.segments) - 1)]

        Lp = seg.start + 0.18 * self.rhat
        Rp = seg.start + 3.85 * self.rhat
        midpoint = 0.5 * (Lp + Rp)

        L_dot = Dot(data_to_manim(Lp), radius=0.064, color=C_CURRENT)
        R_dot = Dot(data_to_manim(Rp), radius=0.064, color=C_CANDIDATE)
        bracket = Line(data_to_manim(Lp), data_to_manim(Rp),
                       color=C_CURRENT, stroke_width=4).set_opacity(0.55)
        probe = Dot(data_to_manim(midpoint), radius=0.095, color=C_TEXT)
        probe_label = make_callout(r"probe", data_to_manim(midpoint),
                                   UP, C_TEXT, max_width=1.4)

        candidate_cell = self.highlight_cell(next_seg.index, C_CANDIDATE, opacity=0.42)
        candidate_site = self.highlight_site(next_seg.index, C_CANDIDATE)

        content = VGroup(
            texm(r"l_{\rm cen}=(L+R)/2",
                 font_size=30, color=C_EQ, max_width=RIGHT_PANEL_WIDTH - 0.7),
            texm(r"\mathbf{r}_{\rm end}=\mathbf{r}+l_{\rm cen}\,\hat{\mathbf{r}}",
                 font_size=30, color=C_EQ, max_width=RIGHT_PANEL_WIDTH - 0.7),
            texm(r"I_{\rm end}=\mathrm{tree\_nearest}(\mathbf{r}_{\rm end},T)",
                 font_size=27, color=C_CANDIDATE,
                 max_width=RIGHT_PANEL_WIDTH - 0.7),
        ).arrange(DOWN, buff=0.32)

        self.right_content = self.replace_right_content(content, self.right_content, run_time=0.70)
        self.play(Create(bracket), FadeIn(L_dot), FadeIn(R_dot), run_time=0.75)
        self.play(FadeIn(probe), FadeIn(probe_label), run_time=0.55)

        # Probe sweeps within the bracket. The label stays clear of the diagram
        # top edge because the bracket midpoint sits in the lower half.
        for t in [0.26, 0.72, 0.50, 0.61]:
            point = (1 - t) * Lp + t * Rp
            self.play(
                probe.animate.move_to(data_to_manim(point)),
                probe_label.animate.next_to(data_to_manim(point), UP, buff=0.12),
                run_time=0.80,
            )

        self.play(FadeIn(candidate_cell), FadeIn(candidate_site), run_time=0.95)
        self.wait(1.35)

        self.candidate_cell = candidate_cell
        self.candidate_site = candidate_site
        self.probe_group = VGroup(L_dot, R_dot, bracket, probe, probe_label)

    # -------------------------------------------------------------------------
    # Scene 6 -- candidate face
    # -------------------------------------------------------------------------

    def scene_face_geometry(self):
        self.next_section("face_geometry", skip_animations=False)
        self.update_algorithm_header(
            "6.~Candidate Voronoi face",
            "The face is the perpendicular bisector between current and candidate sites.",
            2,
        )
        self.play(FadeOut(self.probe_group), run_time=0.50)

        seg = self.segments[self.demo_k]
        next_seg = self.segments[min(self.demo_k + 1, len(self.segments) - 1)]

        xcur = self.points[seg.index]
        xend = self.points[next_seg.index]
        q = xend - xcur
        m = xcur + 0.5 * q
        unit_q = q / np.linalg.norm(q)
        tangent = np.array([-unit_q[1], unit_q[0]])
        face_a = m - 4.0 * tangent
        face_b = m + 4.0 * tangent

        q_arrow = Arrow(data_to_manim(xcur), data_to_manim(xend),
                        buff=0.10, color=C_BLUE, stroke_width=4)
        mid_dot = Dot(data_to_manim(m), radius=0.075, color=C_TEXT)
        face = DashedLine(data_to_manim(face_a), data_to_manim(face_b),
                          color=C_TEXT, stroke_width=3.4)

        # Place the "candidate face" callout on whichever side of the bisector
        # has more empty space, so it doesn't collide with sites or the ray.
        anchor = data_to_manim(m + 1.05 * tangent)
        face_label = make_callout(r"candidate face", anchor,
                                  direction=UP, color=C_TEXT, max_width=2.6)

        content = VGroup(
            texm(r"\mathbf{q}=\mathbf{x}_{\rm end}-\mathbf{x}_{\rm cur}",
                 font_size=30, color=C_BLUE, max_width=RIGHT_PANEL_WIDTH - 0.7),
            texm(r"\mathbf{m}=\mathbf{x}_{\rm cur}+\mathbf{q}/2",
                 font_size=30, color=C_EQ, max_width=RIGHT_PANEL_WIDTH - 0.7),
            texm(r"(\mathbf{x}-\mathbf{m})\cdot\mathbf{q}=0",
                 font_size=32, color=C_OUTPUT, max_width=RIGHT_PANEL_WIDTH - 0.7),
        ).arrange(DOWN, buff=0.34)

        self.right_content = self.replace_right_content(content, self.right_content, run_time=0.70)
        self.play(Create(q_arrow), run_time=0.95)
        self.play(FadeIn(mid_dot), Create(face), FadeIn(face_label), run_time=1.10)
        self.wait(1.50)

        self.face_group = VGroup(q_arrow, mid_dot, face, face_label)

    # -------------------------------------------------------------------------
    # Scene 7 -- intersection math
    # -------------------------------------------------------------------------

    def scene_intersection_math(self):
        self.next_section("intersection_math", skip_animations=False)
        self.update_algorithm_header(
            "7.~Ray-face intersection",
            "Algorithm 2 computes the local distance to the candidate face.",
            3,
        )

        seg = self.segments[self.demo_k]
        accepted = Line(data_to_manim(seg.start), data_to_manim(seg.end),
                        color=C_RAY, stroke_width=8)
        cross_dot = Dot(data_to_manim(seg.end), radius=0.10, color=C_RAY)

        content = VGroup(
            texm(r"\mathbf{c}=\mathbf{m}-\mathbf{r}",
                 font_size=28, color=C_EQ, max_width=RIGHT_PANEL_WIDTH - 0.7),
            texm(r"c_q=\mathbf{c}\cdot\mathbf{q}",
                 font_size=28, color=C_EQ, max_width=RIGHT_PANEL_WIDTH - 0.7),
            texm(r"h_q=\hat{\mathbf{r}}\cdot\mathbf{q}",
                 font_size=28, color=C_EQ, max_width=RIGHT_PANEL_WIDTH - 0.7),
            texm(r"s=c_q/h_q",
                 font_size=34, color=C_OUTPUT, max_width=RIGHT_PANEL_WIDTH - 0.7),
        ).arrange(DOWN, buff=0.28)

        self.right_content = self.replace_right_content(content, self.right_content, run_time=0.70)
        for item in content:
            self.play(Indicate(item, color=C_BLUE, scale_factor=1.04), run_time=0.45)
        self.play(Create(accepted), FadeIn(cross_dot), run_time=1.10)

        bottom = make_bottom_panel("Branch logic")
        branches = VGroup(
            texm(r"h_q=0:\;\text{ignore}", font_size=24, color=C_MUTED),
            texm(r"c_q>0:\;s=c_q/h_q", font_size=24, color=C_EQ),
            texm(r"c_q\leq0,\;h_q>0:\;s=0", font_size=24, color=C_EQ),
            texm(r"\text{else}:\;s=\infty", font_size=24, color=C_MUTED),
        ).arrange(RIGHT, buff=0.55)
        if branches.width > BOTTOM_PANEL_WIDTH - 1.0:
            branches.scale_to_fit_width(BOTTOM_PANEL_WIDTH - 1.0)
        place_in_panel(branches, bottom, y_shift=0.05)

        self.play(FadeIn(bottom), LaggedStartMap(FadeIn, branches, lag_ratio=0.18),
              run_time=1.45)
        self.wait(1.60)

        # ! IMPORTANT: track the accepted segment and the end-of-step dot so we
        # can fade them later in the full-walk transition.
        self.accepted_segment = accepted
        self.cross_dot = cross_dot
        self.branch_group = VGroup(bottom, branches)

    # -------------------------------------------------------------------------
    # Scene 8 -- verify
    # -------------------------------------------------------------------------

    def scene_verify_candidate(self):
        self.next_section("verify_candidate", skip_animations=False)
        self.update_algorithm_header(
            "8.~Verify the candidate crossing",
            "A second nearest-site query catches hidden nearer cells.",
            4,
        )
        self.play(FadeOut(self.branch_group), run_time=0.45)

        seg = self.segments[self.demo_k]
        rcand = Dot(data_to_manim(seg.end + 0.015 * self.rhat),
                    radius=0.095, color=C_RAY)

        content = VGroup(
            texm(r"\mathbf{r}_{\rm cand}=\mathbf{r}+dl_{\rm local}\,\hat{\mathbf{r}}",
                 font_size=27, color=C_EQ, max_width=RIGHT_PANEL_WIDTH - 0.7),
            texm(r"I_{\rm cand}=\mathrm{tree\_nearest}(\mathbf{r}_{\rm cand},T)",
                 font_size=26, color=C_EQ, max_width=RIGHT_PANEL_WIDTH - 0.7),
            texm(r"I_{\rm cand}\in\{\,I_{\rm cur},\,I_{\rm end}\,\}",
                 font_size=29, color=C_OUTPUT, max_width=RIGHT_PANEL_WIDTH - 0.7),
        ).arrange(DOWN, buff=0.34)

        self.right_content = self.replace_right_content(content, self.right_content, run_time=0.70)
        self.play(FadeIn(rcand), run_time=0.60)

        bottom = make_bottom_panel("If the candidate fails")
        fail_text = tex(
            r"Another cell lies in front. Push the candidate and continue.",
            font_size=22, color=C_FAILED, max_width=BOTTOM_PANEL_WIDTH - 0.9,
        )
        place_in_panel(fail_text, bottom, y_shift=0.06)

        # Ghost-cell hint -- placed in the unused upper-right region of the
        # diagram so it doesn't collide with the accepted segment.
        ghost_a = data_to_manim(self.points[8] + np.array([-0.6, -0.55]))
        ghost_b = data_to_manim(self.points[8] + np.array([0.6, 0.55]))
        ghost = DashedLine(ghost_a, ghost_b, color=C_FAILED, stroke_width=3)
        ghost.set_opacity(0.55)

        self.play(FadeIn(bottom), FadeIn(fail_text), Create(ghost), run_time=1.00)
        self.play(Indicate(ghost, color=C_FAILED), run_time=0.90)
        self.wait(1.55)

        self.rcand = rcand
        self.verify_bottom = VGroup(bottom, fail_text)
        self.ghost = ghost

    # -------------------------------------------------------------------------
    # Scene 9 -- save and advance
    # -------------------------------------------------------------------------

    def scene_save_advance(self):
        self.next_section("save_advance", skip_animations=False)
        self.update_algorithm_header(
            "9.~Save and advance",
            "The accepted segment becomes one row in the ray file.",
            5,
        )

        self.play(
            FadeOut(VGroup(
                self.verify_bottom, self.ghost, self.face_group,
                self.current_cell, self.current_site,
                self.candidate_cell, self.candidate_site,
                self.rcand,
            )),
            run_time=0.70,
        )

        seg = self.segments[self.demo_k]
        content = VGroup(
            texm(r"\text{append}\;I_{\rm cur}\rightarrow\mathcal{I}",
                 font_size=28, color=C_CURRENT, max_width=RIGHT_PANEL_WIDTH - 0.7),
            texm(r"\text{append}\;dl_{\rm local}\rightarrow\Delta x",
                 font_size=28, color=C_OUTPUT, max_width=RIGHT_PANEL_WIDTH - 0.7),
            texm(r"\mathbf{r}\leftarrow\mathbf{r}+dl_{\rm local}\,\hat{\mathbf{r}}",
                 font_size=26, color=C_BLUE, max_width=RIGHT_PANEL_WIDTH - 0.7),
            texm(r"I_{\rm cur}\leftarrow I_{\rm end}",
                 font_size=28, color=C_EQ, max_width=RIGHT_PANEL_WIDTH - 0.7),
        ).arrange(DOWN, buff=0.32)
        self.right_content = self.replace_right_content(content, self.right_content, run_time=0.70)

        bottom = make_bottom_panel("One accepted output row")
        row = VGroup(
            tex(r"cell index:", font_size=24, color=C_MUTED),
            tex(rf"\textbf{{{seg.index}}}", font_size=26, color=C_CURRENT),
            tex(r"\quad path length:", font_size=24, color=C_MUTED),
            tex(rf"\textbf{{{seg.length:.2f}}}", font_size=26, color=C_OUTPUT),
        ).arrange(RIGHT, buff=0.30)
        if row.width > BOTTOM_PANEL_WIDTH - 0.8:
            row.scale_to_fit_width(BOTTOM_PANEL_WIDTH - 0.8)
        place_in_panel(row, bottom, y_shift=0.04)

        self.play(FadeIn(bottom), FadeIn(row), run_time=0.70)
        self.play(self.demo_marker.animate.move_to(data_to_manim(seg.end)),
              run_time=1.10)
        self.wait(1.50)

        self.save_bottom = VGroup(bottom, row)

    # -------------------------------------------------------------------------
    # Scene 10 -- full walk
    # -------------------------------------------------------------------------

    def scene_full_walk(self):
        self.next_section("full_walk", skip_animations=False)
        self.update_algorithm_header(
            "10.~Repeat until the final cell",
            "Only accepted segments are saved.",
            5,
        )
        self.play(FadeOut(self.save_bottom), FadeOut(self.right_content),
              FadeOut(self.accepted_segment), FadeOut(self.cross_dot),
              run_time=0.60)

        # Snap the marker back to the ray origin so the full walk animates
        # strictly forward from r_0 rather than starting with a backward jump.
        self.play(
            self.demo_marker.animate.move_to(data_to_manim(self.r0)),
            run_time=0.55,
        )

        # Output table -- placed within the right panel area so it doesn't
        # bleed past the diagram or off the screen edge.
        table_center = RIGHT_PANEL_CENTER + np.array([0.0, -0.10, 0.0])
        table_panel = make_panel(
            RIGHT_PANEL_WIDTH - 1.4, RIGHT_PANEL_HEIGHT - 0.8,
            table_center, title="saved",
        )
        header = VGroup(
            tex(r"$j$", font_size=22, color=C_MUTED),
            tex(r"cell", font_size=22, color=C_CURRENT),
            tex(r"$\Delta x$", font_size=22, color=C_OUTPUT),
        ).arrange(RIGHT, buff=0.55)
        header.next_to(table_panel[1], DOWN, buff=0.26)

        self.play(FadeIn(table_panel), FadeIn(header), run_time=0.70)

        # Progress bar -- INSIDE the bottom panel, not below it.
        bottom = make_bottom_panel("Progress")
        bar_w = BOTTOM_PANEL_WIDTH - 1.6
        bar_y = BOTTOM_PANEL_CENTER[1] - 0.12
        bar_x0 = BOTTOM_PANEL_CENTER[0] - bar_w / 2
        progress_line = Line([bar_x0, bar_y, 0], [bar_x0 + bar_w, bar_y, 0],
                             color=C_DIM, stroke_width=5)
        progress = Line(progress_line.get_start(), progress_line.get_start(),
                        color=C_OUTPUT, stroke_width=5)
        self.play(FadeIn(bottom), Create(progress_line), run_time=0.55)

        saved_rows = VGroup()
        segment_lines = VGroup()
        travelled = 0.0
        max_rows = 5

        for j, seg in enumerate(self.segments):
            cell = self.highlight_cell(seg.index, C_CURRENT, opacity=0.34)
            line = Line(data_to_manim(seg.start), data_to_manim(seg.end),
                        color=C_RAY, stroke_width=7)
            glow = Line(data_to_manim(seg.start), data_to_manim(seg.end),
                        color=C_RAY, stroke_width=13)
            glow.set_opacity(0.18)

            travelled += seg.length
            frac = min(1.0, travelled / self.total_length)
            new_progress = Line(
                progress_line.get_start(),
                progress_line.get_start() + RIGHT * bar_w * frac,
                color=C_OUTPUT, stroke_width=5,
            )

            self.play(FadeIn(cell), run_time=0.25)
            self.play(
                Create(glow), Create(line),
                self.demo_marker.animate.move_to(data_to_manim(seg.end)),
                Transform(progress, new_progress),
                run_time=0.70,
            )
            segment_lines.add(glow, line)

            if j < max_rows:
                row = VGroup(
                    tex(rf"{j+1}", font_size=22, color=C_TEXT),
                    tex(rf"{seg.index}", font_size=22, color=C_CURRENT),
                    tex(rf"{seg.length:.2f}", font_size=22, color=C_OUTPUT),
                ).arrange(RIGHT, buff=0.55)
                row.next_to(header, DOWN, buff=0.22 + 0.32 * j)
                if row.width > table_panel[0].width - 0.30:
                    row.scale_to_fit_width(table_panel[0].width - 0.30)
                saved_rows.add(row)
                self.play(FadeIn(row), FadeOut(cell), run_time=0.28)
            else:
                self.play(FadeOut(cell), run_time=0.16)

        # Wrap up the walk.
        self.wait(0.6)
        self.play(
            FadeOut(VGroup(table_panel, header, saved_rows, bottom,
                           progress_line, progress)),
            run_time=0.60,
        )

        final_bottom = make_bottom_panel("Final geometric ray")
        ids_str = ",".join(str(seg.index) for seg in self.segments[:7])
        dxs_str = ",".join(f"{seg.length:.2f}" for seg in self.segments[:7])
        if len(self.segments) > 7:
            ids_str += r",\ldots"
            dxs_str += r",\ldots"

        arrays = VGroup(
            texm(r"\mathcal{I}=\{" + ids_str + r"\}",
                 font_size=26, color=C_CURRENT, max_width=BOTTOM_PANEL_WIDTH - 1.0),
            texm(r"\Delta x=\{" + dxs_str + r"\}",
                 font_size=26, color=C_OUTPUT, max_width=BOTTOM_PANEL_WIDTH - 1.0),
        ).arrange(DOWN, buff=0.10)
        place_in_panel(arrays, final_bottom, y_shift=0.02)

        self.play(FadeIn(final_bottom), Write(arrays), run_time=1.10)
        self.wait(1.80)

        # ---------------------------------------------------------------
        # CRITICAL CLEANUP: scenes 4-10 share state. We are about to leave
        # the algorithm walk for the physics half of the video, so wipe the
        # walk geometry (ray, segments, marker) and the bottom array panel.
        # The next scene rebuilds the diagram fresh.
        # ---------------------------------------------------------------
        self.play(
            FadeOut(self.base),
            FadeOut(self.demo_marker),
            FadeOut(segment_lines),
            FadeOut(final_bottom),
            FadeOut(arrays),
            run_time=0.90,
        )

    # -------------------------------------------------------------------------
    # Scene 11 -- geometry to physics
    # -------------------------------------------------------------------------

    @subscene
    def scene_geometry_to_physics(self):
        title = scene_heading(
            "11.~Geometry becomes physics",
            "Each path length multiplies a property sampled from the crossed cell.",
        )
        diagram = self.build_diagram(cells_opacity=0.12)
        ray = self.build_ray(color=C_MUTED, width=2.6)

        paths = VGroup()
        for seg in self.segments:
            paths.add(Line(data_to_manim(seg.start), data_to_manim(seg.end),
                           color=C_RAY, stroke_width=6))

        right = make_right_panel("Column contribution")
        eqs = VGroup(
            texm(r"\Delta N_{{\rm ion},j}=n_{{\rm ion},j}\,\Delta x_j",
                 font_size=30, color=C_OUTPUT, max_width=RIGHT_PANEL_WIDTH - 0.7),
            texm(r"N_{\rm ion}=\sum_j n_{{\rm ion},j}\,\Delta x_j",
                 font_size=30, color=C_CURRENT, max_width=RIGHT_PANEL_WIDTH - 0.7),
        ).arrange(DOWN, buff=0.50)
        place_in_panel(eqs, right)

        # Field tags -- only on every second segment, alternating vertical
        # offset, so neighbouring tags don't stack.
        field_tags = VGroup()
        chosen = list(range(0, min(4, len(self.segments)), 2)) + [1]
        chosen = sorted(set(chosen))
        for k, idx in enumerate(chosen[:3]):
            seg = self.segments[idx]
            mid = 0.5 * (seg.start + seg.end)
            tag_str = rf"cell {seg.index}: $n,\,T,\,v$"
            tag_text = tex(tag_str, font_size=18, color=C_TEXT, max_width=1.7)
            tag = label_bg(tag_text, opacity=0.86, buff=0.08)
            vshift = (0.45 if (k % 2 == 0) else -0.45)
            tag.move_to(data_to_manim(mid) + UP * vshift)
            field_tags.add(tag)

        self.play(FadeIn(title), FadeIn(diagram), FadeIn(ray), run_time=0.90)
        self.play(Create(paths), run_time=1.50)
        self.play(FadeIn(field_tags, lag_ratio=0.12), run_time=0.95)
        self.play(FadeIn(right), Write(eqs), run_time=1.10)
        self.wait(1.70)

    # -------------------------------------------------------------------------
    # Scene 12 -- spectrum
    # -------------------------------------------------------------------------

    @subscene
    def scene_spectrum(self):
        title = scene_heading(
            "12.~From ray file to spectrum",
            "Velocity shifts move components in wavelength; optical depths add.",
        )

        top_panel = make_panel(12.85, 1.20, [0.10, 2.15, 0], title="Spectral bookkeeping")
        eqs = VGroup(
            texm(r"z_{\rm dop}=v_{\rm los}/c",
                 font_size=28, color=C_BLUE),
            texm(r"z_{\rm eff}=(1+z_{\rm cosmo})(1+z_{\rm dop})-1",
                 font_size=28, color=C_CURRENT),
            texm(r"F=e^{-\tau}", font_size=30, color=C_OUTPUT),
        ).arrange(RIGHT, buff=0.85)
        if eqs.width > top_panel[0].width - 0.7:
            eqs.scale_to_fit_width(top_panel[0].width - 0.7)
        place_in_panel(eqs, top_panel, y_shift=0.04)

        axes = Axes(
            x_range=[-500, 500, 250],
            y_range=[0, 1.12, 0.5],
            x_length=11.0, y_length=3.30,
            tips=False,
            axis_config={"color": C_PANEL_EDGE, "stroke_width": 2,
                         "include_ticks": True},
        ).move_to([0.10, -0.90, 0])

        xlabel = tex(r"velocity [km/s]", font_size=24, color=C_MUTED)
        xlabel.next_to(axes, DOWN, buff=0.22)
        ylabel = tex(r"normalized flux", font_size=24, color=C_MUTED).rotate(PI / 2)
        ylabel.next_to(axes, LEFT, buff=0.22)

        x_vals = np.linspace(-500, 500, 750)
        components = [(-190, 0.42, 55), (15, 0.70, 42), (205, 0.30, 74)]

        def curve_for(n, color):
            tau = np.zeros_like(x_vals)
            for center, depth, width in components[:n]:
                tau0 = -math.log(max(1e-3, 1.0 - depth))
                tau += tau0 * np.exp(-0.5 * ((x_vals - center) / width) ** 2)
            flux = np.exp(-tau)
            curve = VMobject(color=color, stroke_width=4)
            curve.set_points_as_corners([axes.c2p(float(x), float(y))
                                         for x, y in zip(x_vals, flux)])
            return curve

        self.play(FadeIn(title), FadeIn(top_panel), Write(eqs), run_time=1.00)
        self.play(Create(axes), FadeIn(xlabel), FadeIn(ylabel), run_time=1.00)

        curve = curve_for(1, C_BLUE)
        self.play(Create(curve), run_time=1.10)
        self.wait(0.35)
        self.play(Transform(curve, curve_for(2, C_PURPLE)), run_time=1.05)
        self.wait(0.35)
        self.play(Transform(curve, curve_for(3, C_RAY)), run_time=1.15)
        self.wait(1.65)

    # -------------------------------------------------------------------------
    # Scene 13 -- compare LightRay
    # -------------------------------------------------------------------------

    @subscene
    def scene_compare_lightray(self):
        title = scene_heading(
            "13.~Relation to yt / Trident LightRay",
            "The spectrum engine can be reused; the ray sampling is different.",
        )

        left = make_panel(6.05, 3.20, [-3.55, 0.10, 0], title="yt / Trident LightRay")
        left_text = tex_para([
            r"general yt ray sampling",
            r"Arepo handled as SPH-like",
            r"smoothing-length contributions",
            r"not expected to match exactly",
        ], font_size=26, color=C_TEXT, max_width=5.3)
        place_in_panel(left_text, left, y_shift=0.05)

        right = make_panel(6.05, 3.20, [3.55, 0.10, 0], title="SALSA-style meshless")
        right_text = tex_para([
            r"Voronoi generating sites",
            r"true geometric path lengths",
            r"no full mesh stored",
            r"output is yt-compatible",
        ], font_size=26, color=C_TEXT, max_width=5.3)
        place_in_panel(right_text, right, y_shift=0.05)

        bottom = make_bottom_panel()
        msg = tex(
            r"Same spectrum machinery, different ray-sampling assumptions.",
            font_size=26, color=C_CURRENT, max_width=BOTTOM_PANEL_WIDTH - 0.9,
        )
        place_in_panel(msg, bottom, y_shift=0.02)

        self.play(FadeIn(title), run_time=0.60)
        self.play(FadeIn(left), FadeIn(left_text), run_time=0.95)
        self.play(FadeIn(right), FadeIn(right_text), run_time=0.95)
        self.play(FadeIn(bottom), FadeIn(msg), run_time=0.70)
        self.wait(1.70)

    # -------------------------------------------------------------------------
    # Scene 14 -- summary
    # -------------------------------------------------------------------------

    @subscene
    def scene_summary(self):
        title = scene_heading(
            "Summary",
            "Nearest queries and face intersections produce the ray file.",
        )
        card = make_panel(12.0, 4.35, [0.0, -0.30, 0])

        bullets = VGroup(
            tex(r"1.\quad the nearest-neighbour tree identifies the current cell",
                font_size=30, color=C_TEXT, max_width=10.8),
            tex(r"2.\quad bisection and a face intersection find the next crossing",
                font_size=30, color=C_TEXT, max_width=10.8),
            tex(r"3.\quad ordered cell IDs and path lengths feed the spectrum",
                font_size=30, color=C_TEXT, max_width=10.8),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.42)
        bullets.move_to(card[0].get_center() + UP * 0.55)

        flow = texm(
            r"\{\mathbf{x}_i\},\;\mathbf{r}_0,\;\hat{\mathbf{r}},\;S"
            r"\;\longrightarrow\;(\mathcal{I},\,\Delta x)"
            r"\;\longrightarrow\;F(\lambda)",
            font_size=34, color=C_CURRENT, max_width=11.0,
        )
        flow.next_to(bullets, DOWN, buff=0.65)

        self.play(FadeIn(title), FadeIn(card), run_time=0.70)
        self.play(LaggedStartMap(FadeIn, bullets, lag_ratio=0.20), run_time=1.45)
        self.play(Write(flow), run_time=1.20)
        self.wait(2.10)