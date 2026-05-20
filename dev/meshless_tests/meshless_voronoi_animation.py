"""
Production educational Manim animation for SALSA-style meshless Voronoi
ray tracing.

Design goals:
    - Pure black background.
    - LaTeX / Computer Modern typography.
    - No persistent ray/object across unrelated scenes.
    - Left: diagram safe zone.
    - Right: equations/concepts safe zone.
    - Bottom: output/progress safe zone.
    - One idea per scene.
    - Slow enough for human comprehension.

Render from repository root:

Preview:
    /Users/wavefunction/github_repos/m61-tng/.venv/bin/python -m manim -pql \
        dev/meshless_tests/meshless_voronoi_animation.py SalsaMeshlessVoronoiRayTracing

High quality:
    /Users/wavefunction/github_repos/m61-tng/.venv/bin/python -m manim -pqh \
        dev/meshless_tests/meshless_voronoi_animation.py SalsaMeshlessVoronoiRayTracing

Save named sections:
    /Users/wavefunction/github_repos/m61-tng/.venv/bin/python -m manim -pql --save_sections \
        dev/meshless_tests/meshless_voronoi_animation.py SalsaMeshlessVoronoiRayTracing
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import Voronoi

from manim import *
from manim.utils.color import ManimColor


# =============================================================================
# Render configuration
# =============================================================================

config.pixel_width = 1920
config.pixel_height = 1080
config.frame_rate = 30
config.background_color = "#000000"
config.renderer = "cairo"

TEX_TEMPLATE = TexTemplate()
TEX_TEMPLATE.add_to_preamble(
    r"""
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}
\usepackage{physics}
"""
)
config.tex_template = TEX_TEMPLATE


# =============================================================================
# Palette
# =============================================================================

BLACK = "#000000"
WHITE_SOFT = "#f4f4f4"
GREY = "#9aa3ad"
GREY_DARK = "#20252f"

BLUE = "#58c4dd"
BLUE_DIM = "#2d7f95"
YELLOW = "#ffd166"
RED = "#ff4d6d"
ORANGE = "#fb8500"
GREEN = "#06d6a0"
PURPLE = "#b998ff"
PANEL = "#0b0f19"
PANEL_EDGE = "#2f4058"

CELL_COLORS = [
    "#143a52",
    "#1d3557",
    "#22577a",
    "#2a6f97",
    "#355070",
    "#1b4965",
    "#386641",
    "#0b525b",
    "#4d194d",
    "#3c096c",
    "#144552",
    "#284b63",
    "#31572c",
    "#4a4e69",
]


# =============================================================================
# Layout constants
# =============================================================================

DATA_XMIN, DATA_XMAX = -6.4, 6.4
DATA_YMIN, DATA_YMAX = -3.9, 4.4

DIAGRAM_CENTER = np.array([-3.05, -0.15, 0.0])
DIAGRAM_WIDTH = 6.15
DIAGRAM_HEIGHT = 4.35

DATA_SCALE = min(
    DIAGRAM_WIDTH / (DATA_XMAX - DATA_XMIN),
    DIAGRAM_HEIGHT / (DATA_YMAX - DATA_YMIN),
)

RIGHT_CENTER = np.array([3.65, -0.10, 0.0])
RIGHT_WIDTH = 5.55
RIGHT_HEIGHT = 4.65

BOTTOM_CENTER = np.array([0.0, -3.30, 0.0])
BOTTOM_WIDTH = 11.80
BOTTOM_HEIGHT = 0.95

TITLE_Y = 3.55
SUBTITLE_Y = 3.17


# =============================================================================
# Timing
# =============================================================================

TINY = 0.25
FAST = 0.45
MED = 0.75
SLOW = 1.10
PAUSE = 1.20


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
# Geometry utilities
# =============================================================================

def finite_voronoi_polygons_2d(vor: Voronoi, radius: float = 80.0):
    """Finite 2D Voronoi polygons for visualization only."""
    if vor.points.shape[1] != 2:
        raise ValueError("Only 2D point sets are supported.")

    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)

    all_ridges = {}
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


def clip_polygon_to_box(poly: np.ndarray, xmin: float, xmax: float, ymin: float, ymax: float):
    """Clip polygon to rectangular domain."""

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


def ray_face_intersection(
    r: np.ndarray,
    rhat: np.ndarray,
    xcur: np.ndarray,
    xend: np.ndarray,
    dl_local: float = np.inf,
) -> float:
    """SALSA Algorithm 2 branch logic."""
    q = xend - xcur
    m = xcur + 0.5 * q
    c = m - r

    c_q = float(np.dot(c, q))
    h_q = float(np.dot(rhat, q))

    if abs(h_q) <= 1e-12:
        return dl_local

    if c_q > 0:
        s = c_q / h_q
    elif h_q > 0:
        s = 0.0
    else:
        s = np.inf

    if 0.0 <= s <= dl_local:
        return float(s)
    return dl_local


def compute_meshless_path(points: np.ndarray, r0: np.ndarray, r1: np.ndarray) -> list[RaySegment]:
    """
    Stable educational Voronoi walk.

    This checks all possible candidate sites for the nearest valid face.
    The real SALSA algorithm accelerates candidate discovery using a tree and
    bisection. The animation explains that tree/bisection logic separately.
    """
    delta = r1 - r0
    total = float(np.linalg.norm(delta))
    rhat = delta / total

    r = r0.copy()
    travelled = 0.0

    current = nearest_index(points, r)
    final = nearest_index(points, r1)
    segments = []

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


def data_to_manim(p: np.ndarray) -> np.ndarray:
    return DIAGRAM_CENTER + np.array([DATA_SCALE * p[0], DATA_SCALE * p[1], 0.0])


def color_mix(a: str, b: str, alpha: float):
    return ManimColor(a).interpolate(ManimColor(b), alpha)


# =============================================================================
# LaTeX / layout helpers
# =============================================================================

def tex_text(text: str, size: int = 28, color: str = WHITE_SOFT, max_width: float | None = None):
    """
    Text rendered through LaTeX, not system fonts.
    Use short strings only.
    """
    mob = Tex(r"\textrm{" + text + "}", font_size=size, color=color, tex_template=TEX_TEMPLATE)
    if max_width is not None and mob.width > max_width:
        mob.scale_to_fit_width(max_width)
    return mob


def tex_bold(text: str, size: int = 32, color: str = WHITE_SOFT, max_width: float | None = None):
    mob = Tex(r"\textbf{" + text + "}", font_size=size, color=color, tex_template=TEX_TEMPLATE)
    if max_width is not None and mob.width > max_width:
        mob.scale_to_fit_width(max_width)
    return mob


def mtex(tex: str, size: int = 34, color: str = WHITE_SOFT, max_width: float | None = None):
    mob = MathTex(tex, font_size=size, color=color, tex_template=TEX_TEMPLATE)
    if max_width is not None and mob.width > max_width:
        mob.scale_to_fit_width(max_width)
    return mob


def panel(width: float, height: float, center, title: str | None = None):
    box = RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.12,
        stroke_width=1.3,
        stroke_color=PANEL_EDGE,
        fill_color=PANEL,
        fill_opacity=0.96,
    ).move_to(center)

    group = VGroup(box)

    if title:
        t = tex_bold(title, size=21, color=YELLOW, max_width=width - 0.35)
        t.move_to(box.get_top() + DOWN * 0.25)
        group.add(t)

    return group


def right_panel(title: str | None = None):
    return panel(RIGHT_WIDTH, RIGHT_HEIGHT, RIGHT_CENTER, title)


def bottom_panel(title: str | None = None):
    return panel(BOTTOM_WIDTH, BOTTOM_HEIGHT, BOTTOM_CENTER, title)


def put_in_panel(obj: Mobject, p: VGroup, y_shift: float = 0.05):
    obj.move_to(p[0].get_center() + DOWN * y_shift)
    return obj


def heading(title: str, subtitle: str | None = None):
    t = tex_bold(title, size=34, color=WHITE_SOFT, max_width=12.0)
    t.move_to([0.0, TITLE_Y, 0.0])
    g = VGroup(t)
    if subtitle:
        s = tex_text(subtitle, size=18, color=GREY, max_width=12.0)
        s.move_to([0.0, SUBTITLE_Y, 0.0])
        g.add(s)
    return g


def label_box(obj: Mobject, opacity: float = 0.88, buff: float = 0.08):
    return VGroup(
        BackgroundRectangle(obj, color=PANEL, fill_opacity=opacity, buff=buff),
        obj,
    )


def callout(text: str, at: np.ndarray, direction=UP, color=WHITE_SOFT):
    t = tex_text(text, size=15, color=color, max_width=1.9)
    b = label_box(t, opacity=0.90)
    b.next_to(at, direction, buff=0.10)
    return b


def vstack(items: list[Mobject], buff: float = 0.30, aligned_edge=ORIGIN):
    g = VGroup(*items)
    if aligned_edge is ORIGIN:
        g.arrange(DOWN, buff=buff)
    else:
        g.arrange(DOWN, aligned_edge=aligned_edge, buff=buff)
    return g


def step_bar(current: int, labels: list[str]):
    width = RIGHT_WIDTH - 0.45
    y = 2.50
    x0 = RIGHT_CENTER[0] - width / 2
    spacing = width / (len(labels) - 1)

    base = Line([x0, y, 0], [x0 + width, y, 0], color=GREY_DARK, stroke_width=2)

    dots = VGroup()
    for i in range(len(labels)):
        if i < current:
            c, op = GREEN, 1.0
        elif i == current:
            c, op = YELLOW, 1.0
        else:
            c, op = GREY, 0.35
        d = Dot([x0 + i * spacing, y, 0], radius=0.060, color=c)
        d.set_opacity(op)
        dots.add(d)

    lab = tex_text(labels[current], size=15, color=YELLOW, max_width=1.4)
    lab.next_to(dots[current], UP, buff=0.10)

    return VGroup(base, dots, lab)


# =============================================================================
# Main scene
# =============================================================================

class SalsaMeshlessVoronoiRayTracing(Scene):
    STEP_LABELS = ["nearest", "probe", "face", "intersect", "verify", "save"]

    def construct(self):
        self.prepare()

        self.next_section("00_title", skip_animations=False)
        self.scene_title()

        self.next_section("01_ownership", skip_animations=False)
        self.scene_ownership()

        self.next_section("02_ray_problem", skip_animations=False)
        self.scene_ray_problem()

        self.next_section("03_meshless", skip_animations=False)
        self.scene_meshless()

        self.next_section("04_nearest", skip_animations=False)
        self.scene_nearest()

        self.next_section("05_probe", skip_animations=False)
        self.scene_probe()

        self.next_section("06_face", skip_animations=False)
        self.scene_face()

        self.next_section("07_intersect", skip_animations=False)
        self.scene_intersect()

        self.next_section("08_verify", skip_animations=False)
        self.scene_verify()

        self.next_section("09_save", skip_animations=False)
        self.scene_save()

        self.next_section("10_repeat", skip_animations=False)
        self.scene_repeat()

        self.next_section("11_physics", skip_animations=False)
        self.scene_physics()

        self.next_section("12_spectrum", skip_animations=False)
        self.scene_spectrum()

        self.next_section("13_compare", skip_animations=False)
        self.scene_compare()

        self.next_section("14_summary", skip_animations=False)
        self.scene_summary()

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    def prepare(self):
        self.points = np.array(
            [
                [-5.5, -2.6],
                [-4.2, 1.9],
                [-2.7, -0.8],
                [-1.45, 2.8],
                [0.0, -2.25],
                [0.95, 0.65],
                [2.25, 2.85],
                [3.2, -1.35],
                [4.75, 1.0],
                [5.25, -2.75],
                [-0.35, 3.85],
                [-5.8, 3.55],
                [5.55, 3.70],
                [-3.8, -3.45],
            ],
            dtype=float,
        )

        self.r0 = np.array([-6.05, -3.20])
        self.r1 = np.array([5.75, 3.20])
        self.rhat = (self.r1 - self.r0) / np.linalg.norm(self.r1 - self.r0)
        self.total_length = float(np.linalg.norm(self.r1 - self.r0))

        self.segments = compute_meshless_path(self.points, self.r0, self.r1)
        self.demo_k = min(2, max(0, len(self.segments) - 2))

        vor = Voronoi(self.points)
        self.regions, self.vertices = finite_voronoi_polygons_2d(vor)

    # -------------------------------------------------------------------------
    # Draw primitives
    # -------------------------------------------------------------------------

    def boundary(self):
        return Rectangle(
            width=DIAGRAM_WIDTH,
            height=DIAGRAM_HEIGHT,
            stroke_color=PANEL_EDGE,
            stroke_width=1.3,
        ).move_to(DIAGRAM_CENTER)

    def cells(self, opacity: float = 0.22):
        group = VGroup()
        for i, region in enumerate(self.regions):
            poly = self.vertices[region]
            clipped = clip_polygon_to_box(poly, DATA_XMIN, DATA_XMAX, DATA_YMIN, DATA_YMAX)
            if clipped is None:
                group.add(VMobject())
                continue
            m = Polygon(*[data_to_manim(p) for p in clipped])
            m.set_fill(CELL_COLORS[i % len(CELL_COLORS)], opacity=opacity)
            m.set_stroke("#8fa9c2", width=0.75, opacity=0.52)
            group.add(m)
        return group

    def sites(self, radius=0.048):
        return VGroup(*[Dot(data_to_manim(p), radius=radius, color="#f4efc3") for p in self.points])

    def diagram(self, cell_opacity=0.18, note=True):
        self._cells = self.cells(cell_opacity)
        self._sites = self.sites()
        b = self.boundary()

        g = VGroup(self._cells, b, self._sites)

        if note:
            n = tex_text("Voronoi overlay for explanation only", size=13, color=GREY, max_width=DIAGRAM_WIDTH)
            n = label_box(n, opacity=0.86)
            n.move_to(DIAGRAM_CENTER + DOWN * (DIAGRAM_HEIGHT / 2 + 0.23))
            g.add(n)

        return g

    def ray(self, color=GREY, width=3.0):
        arr = Arrow(
            data_to_manim(self.r0),
            data_to_manim(self.r1),
            buff=0,
            color=color,
            stroke_width=width,
            max_tip_length_to_length_ratio=0.025,
        )
        s = Dot(data_to_manim(self.r0), radius=0.075, color=BLUE)
        e = Dot(data_to_manim(self.r1), radius=0.075, color=YELLOW)
        return VGroup(arr, s, e)

    def highlight_cell(self, index: int, color: str, opacity=0.45):
        c = self._cells[index].copy()
        c.set_fill(color, opacity=opacity)
        c.set_stroke(color, width=2.8, opacity=1.0)
        return c

    def highlight_site(self, index: int, color: str):
        return Dot(data_to_manim(self.points[index]), radius=0.095, color=color)

    def clear_scene(self):
        if self.mobjects:
            self.play(*[FadeOut(m) for m in list(self.mobjects)], run_time=0.55)

    # -------------------------------------------------------------------------
    # Scene 0
    # -------------------------------------------------------------------------

    def scene_title(self):
        title = tex_bold("Meshless Voronoi Ray Tracing", size=58, color=WHITE_SOFT, max_width=11.5)
        subtitle = tex_text(
            "Finding gas-cell intersections without constructing the full mesh",
            size=26,
            color=GREY,
            max_width=11.5,
        )
        tag = tex_text(
            "SALSA-style sightline generation for synthetic spectra",
            size=22,
            color=BLUE,
            max_width=11.5,
        )

        group = VGroup(title, subtitle, tag).arrange(DOWN, buff=0.32).move_to(ORIGIN)

        self.play(FadeIn(title, shift=DOWN * 0.18), run_time=0.90)
        self.play(FadeIn(subtitle, shift=DOWN * 0.12), run_time=0.75)
        self.play(FadeIn(tag), run_time=0.65)
        self.wait(1.25)
        self.clear_scene()

    # -------------------------------------------------------------------------
    # Scene 1
    # -------------------------------------------------------------------------

    def scene_ownership(self):
        h = heading(
            "1. Voronoi ownership",
            "A spatial point belongs to the nearest generating site.",
        )
        d = self.diagram(cell_opacity=0.30)

        rp = right_panel("Definition")
        eq = mtex(
            r"V_i=\{\mathbf{x}:\|\mathbf{x}-\mathbf{x}_i\|"
            r"\leq\|\mathbf{x}-\mathbf{x}_j\|,\;j\neq i\}",
            size=27,
            color=WHITE_SOFT,
            max_width=RIGHT_WIDTH - 0.65,
        )
        line1 = tex_text("Shown for explanation.", size=19, color=GREY)
        line2 = tex_text("Not stored by the algorithm.", size=19, color=GREY)
        content = vstack([eq, line1, line2], buff=0.34)
        put_in_panel(content, rp, y_shift=0.05)

        focus = 5
        hc = self.highlight_cell(focus, YELLOW, opacity=0.55)
        hs = self.highlight_site(focus, YELLOW)
        lab = callout("one generator", data_to_manim(self.points[focus]), UP, YELLOW)

        self.play(FadeIn(h), run_time=FAST)
        self.play(FadeIn(d), run_time=SLOW)
        self.play(FadeIn(rp), Write(eq), FadeIn(line1), FadeIn(line2), run_time=SLOW)
        self.play(FadeIn(hc), FadeIn(hs), FadeIn(lab), run_time=MED)
        self.wait(PAUSE)
        self.clear_scene()

    # -------------------------------------------------------------------------
    # Scene 2
    # -------------------------------------------------------------------------

    def scene_ray_problem(self):
        h = heading(
            "2. The ray-tracing problem",
            "Find which cells are crossed, and the distance inside each cell.",
        )
        d = self.diagram(cell_opacity=0.16)
        r = self.ray(color=WHITE_SOFT, width=4.0)

        start = label_box(mtex(r"\mathbf{r}_0", size=22, color=BLUE), opacity=0.86)
        start.next_to(r[1], DL, buff=0.08)
        end = label_box(mtex(r"\mathbf{r}(S)", size=22, color=YELLOW), opacity=0.86)
        end.next_to(r[2], UR, buff=0.08)

        rp = right_panel("Ray")
        eq1 = mtex(r"\mathbf{r}(s)=\mathbf{r}_0+s\hat{\mathbf{r}}", size=34, max_width=RIGHT_WIDTH - 0.6)
        eq2 = mtex(r"0\leq s\leq S", size=31, color=GREY, max_width=RIGHT_WIDTH - 0.6)
        out1 = mtex(r"\mathcal{I}=\{?,?,?,\ldots\}", size=31, color=GREEN)
        out2 = mtex(r"\Delta x=\{?,?,?,\ldots\}", size=31, color=GREEN)
        content = vstack([eq1, eq2, out1, out2], buff=0.35)
        put_in_panel(content, rp)

        bp = bottom_panel("Output")
        msg = tex_text(
            "Geometry first: cell IDs and path lengths. Gas physics comes afterward.",
            size=21,
            color=WHITE_SOFT,
            max_width=BOTTOM_WIDTH - 0.6,
        )
        put_in_panel(msg, bp, y_shift=0.05)

        self.play(FadeIn(h), FadeIn(d), run_time=MED)
        self.play(Create(r[0]), FadeIn(r[1:]), FadeIn(start), FadeIn(end), run_time=SLOW)
        self.play(FadeIn(rp), LaggedStartMap(Write, content, lag_ratio=0.16), run_time=1.45)
        self.play(FadeIn(bp), FadeIn(msg), run_time=MED)
        self.wait(PAUSE)
        self.clear_scene()

    # -------------------------------------------------------------------------
    # Scene 3
    # -------------------------------------------------------------------------

    def scene_meshless(self):
        h = heading(
            "3. Meshless strategy",
            "The global Voronoi mesh is not built or stored.",
        )

        left = panel(5.25, 3.15, [-3.20, 0.05, 0], "Explicit mesh route")
        left_lines = vstack(
            [
                tex_text("build all cells", size=22),
                tex_text("store neighbours", size=22),
                tex_text("store faces", size=22),
                tex_text("walk face table", size=22),
            ],
            buff=0.23,
        )
        put_in_panel(left_lines, left, y_shift=0.07)
        cross = Cross(left[0], stroke_width=5, color=RED)

        right = panel(5.25, 3.15, [3.20, 0.05, 0], "SALSA route")
        right_lines = vstack(
            [
                tex_text("nearest-neighbor tree", size=22),
                tex_text("site positions", size=22),
                mtex(r"\mathrm{tree\_nearest}(\mathbf{r},T)", size=25, color=GREEN),
                tex_text("bisection along ray", size=22),
            ],
            buff=0.23,
        )
        put_in_panel(right_lines, right, y_shift=0.07)

        bp = bottom_panel("Inputs")
        inp = mtex(
            r"T,\;\{\mathbf{x}_i\},\;\mathbf{r}_0,\;\hat{\mathbf{r}},\;S",
            size=32,
            color=BLUE,
            max_width=BOTTOM_WIDTH - 0.8,
        )
        put_in_panel(inp, bp, y_shift=0.05)

        self.play(FadeIn(h), run_time=FAST)
        self.play(FadeIn(left), FadeIn(left_lines), run_time=MED)
        self.play(Create(cross), run_time=FAST)
        self.play(FadeIn(right), FadeIn(right_lines), run_time=MED)
        self.play(FadeIn(bp), Write(inp), run_time=MED)
        self.wait(PAUSE)
        self.clear_scene()

    # -------------------------------------------------------------------------
    # Algorithm base
    # -------------------------------------------------------------------------

    def algorithm_base(self, title: str, subtitle: str, step: int):
        h = heading(title, subtitle)
        d = self.diagram(cell_opacity=0.15)
        r = self.ray(color=GREY, width=3.0)
        rp = right_panel()
        sb = step_bar(step, self.STEP_LABELS)

        self.play(FadeIn(h), FadeIn(d), FadeIn(r), FadeIn(rp), FadeIn(sb), run_time=MED)
        return h, d, r, rp, sb

    # -------------------------------------------------------------------------
    # Scene 4
    # -------------------------------------------------------------------------

    def scene_nearest(self):
        h, d, r, rp, sb = self.algorithm_base(
            "4. Current cell",
            "At the current ray point, query the nearest generator.",
            0,
        )

        seg = self.segments[self.demo_k]
        marker = Dot(data_to_manim(seg.start), radius=0.10, color=BLUE)
        hc = self.highlight_cell(seg.index, YELLOW, opacity=0.48)
        hs = self.highlight_site(seg.index, YELLOW)
        link = DashedLine(data_to_manim(seg.start), data_to_manim(self.points[seg.index]), color=YELLOW, stroke_width=2.0)

        eqs = vstack(
            [
                mtex(r"I_{\rm cur}=\mathrm{tree\_nearest}(\mathbf{r},T)", size=28, max_width=RIGHT_WIDTH - 0.65),
                mtex(r"\mathbf{x}_{\rm cur}=\mathbf{x}_{I_{\rm cur}}", size=28, color=YELLOW, max_width=RIGHT_WIDTH - 0.65),
            ],
            buff=0.45,
        )
        put_in_panel(eqs, rp)

        self.play(FadeIn(eqs), run_time=MED)
        self.play(FadeIn(marker), run_time=FAST)
        self.play(FadeIn(hc), FadeIn(hs), Create(link), run_time=SLOW)
        self.wait(PAUSE)
        self.clear_scene()

    # -------------------------------------------------------------------------
    # Scene 5
    # -------------------------------------------------------------------------

    def scene_probe(self):
        h, d, r, rp, sb = self.algorithm_base(
            "5. Look ahead",
            "Bisection searches for a point owned by a different generator.",
            1,
        )

        seg = self.segments[self.demo_k]
        nxt = self.segments[min(self.demo_k + 1, len(self.segments) - 1)]

        marker = Dot(data_to_manim(seg.start), radius=0.10, color=BLUE)
        hc = self.highlight_cell(seg.index, YELLOW, opacity=0.42)
        hs = self.highlight_site(seg.index, YELLOW)
        cc = self.highlight_cell(nxt.index, RED, opacity=0.42)
        cs = self.highlight_site(nxt.index, RED)

        Lp = seg.start + 0.18 * self.rhat
        Rp = seg.start + 3.85 * self.rhat
        mid = 0.5 * (Lp + Rp)

        bracket = Line(data_to_manim(Lp), data_to_manim(Rp), color=YELLOW, stroke_width=4).set_opacity(0.55)
        ld = Dot(data_to_manim(Lp), radius=0.055, color=YELLOW)
        rd = Dot(data_to_manim(Rp), radius=0.055, color=RED)
        probe = Dot(data_to_manim(mid), radius=0.09, color=WHITE_SOFT)
        probe_lab = callout("probe", data_to_manim(mid), UP, WHITE_SOFT)

        eqs = vstack(
            [
                mtex(r"l_{\rm cen}=(L+R)/2", size=27, max_width=RIGHT_WIDTH - 0.65),
                mtex(r"\mathbf{r}_{\rm end}=\mathbf{r}+l_{\rm cen}\hat{\mathbf{r}}", size=26, max_width=RIGHT_WIDTH - 0.65),
                mtex(r"I_{\rm end}=\mathrm{tree\_nearest}(\mathbf{r}_{\rm end},T)", size=24, color=RED, max_width=RIGHT_WIDTH - 0.65),
            ],
            buff=0.34,
        )
        put_in_panel(eqs, rp)

        self.play(FadeIn(eqs), FadeIn(marker), FadeIn(hc), FadeIn(hs), run_time=MED)
        self.play(Create(bracket), FadeIn(ld), FadeIn(rd), FadeIn(probe), FadeIn(probe_lab), run_time=MED)

        for t in [0.25, 0.72, 0.50, 0.61]:
            p = (1 - t) * Lp + t * Rp
            self.play(
                probe.animate.move_to(data_to_manim(p)),
                probe_lab.animate.next_to(data_to_manim(p), UP, buff=0.10),
                run_time=0.70,
            )

        self.play(FadeIn(cc), FadeIn(cs), run_time=MED)
        self.wait(PAUSE)
        self.clear_scene()

    # -------------------------------------------------------------------------
    # Scene 6
    # -------------------------------------------------------------------------

    def scene_face(self):
        h, d, r, rp, sb = self.algorithm_base(
            "6. Candidate face",
            "The boundary between two cells is a perpendicular bisector.",
            2,
        )

        seg = self.segments[self.demo_k]
        nxt = self.segments[min(self.demo_k + 1, len(self.segments) - 1)]

        hc = self.highlight_cell(seg.index, YELLOW, opacity=0.36)
        hs = self.highlight_site(seg.index, YELLOW)
        cc = self.highlight_cell(nxt.index, RED, opacity=0.36)
        cs = self.highlight_site(nxt.index, RED)

        xcur = self.points[seg.index]
        xend = self.points[nxt.index]
        q = xend - xcur
        m = xcur + 0.5 * q
        uq = q / np.linalg.norm(q)
        tangent = np.array([-uq[1], uq[0]])

        face_a = m - 4.2 * tangent
        face_b = m + 4.2 * tangent

        q_arrow = Arrow(data_to_manim(xcur), data_to_manim(xend), buff=0.10, color=BLUE, stroke_width=4)
        mdot = Dot(data_to_manim(m), radius=0.075, color=WHITE_SOFT)
        face = DashedLine(data_to_manim(face_a), data_to_manim(face_b), color=WHITE_SOFT, stroke_width=3.2)
        flab = callout("candidate face", data_to_manim(m + 0.85 * tangent), UP, WHITE_SOFT)

        eqs = vstack(
            [
                mtex(r"\mathbf{q}=\mathbf{x}_{\rm end}-\mathbf{x}_{\rm cur}", size=27, color=BLUE, max_width=RIGHT_WIDTH - 0.65),
                mtex(r"\mathbf{m}=\mathbf{x}_{\rm cur}+\mathbf{q}/2", size=27, max_width=RIGHT_WIDTH - 0.65),
                mtex(r"(\mathbf{x}-\mathbf{m})\cdot\mathbf{q}=0", size=29, color=GREEN, max_width=RIGHT_WIDTH - 0.65),
            ],
            buff=0.36,
        )
        put_in_panel(eqs, rp)

        self.play(FadeIn(eqs), FadeIn(hc), FadeIn(hs), FadeIn(cc), FadeIn(cs), run_time=MED)
        self.play(Create(q_arrow), run_time=MED)
        self.play(FadeIn(mdot), Create(face), FadeIn(flab), run_time=SLOW)
        self.wait(PAUSE)
        self.clear_scene()

    # -------------------------------------------------------------------------
    # Scene 7
    # -------------------------------------------------------------------------

    def scene_intersect(self):
        h, d, r, rp, sb = self.algorithm_base(
            "7. Ray-face intersection",
            "Algorithm 2 gives the distance to the candidate face.",
            3,
        )

        seg = self.segments[self.demo_k]
        nxt = self.segments[min(self.demo_k + 1, len(self.segments) - 1)]

        hc = self.highlight_cell(seg.index, YELLOW, opacity=0.30)
        cc = self.highlight_cell(nxt.index, RED, opacity=0.30)

        xcur = self.points[seg.index]
        xend = self.points[nxt.index]
        q = xend - xcur
        m = xcur + 0.5 * q
        uq = q / np.linalg.norm(q)
        tangent = np.array([-uq[1], uq[0]])
        face = DashedLine(data_to_manim(m - 4.2 * tangent), data_to_manim(m + 4.2 * tangent), color=WHITE_SOFT, stroke_width=3)

        current_ray = Line(data_to_manim(seg.start), data_to_manim(seg.end), color=BLUE, stroke_width=8)
        crossing = Dot(data_to_manim(seg.end), radius=0.10, color=BLUE)

        eqs = vstack(
            [
                mtex(r"\mathbf{c}=\mathbf{m}-\mathbf{r}", size=25, max_width=RIGHT_WIDTH - 0.65),
                mtex(r"c_q=\mathbf{c}\cdot\mathbf{q}", size=25, max_width=RIGHT_WIDTH - 0.65),
                mtex(r"h_q=\hat{\mathbf{r}}\cdot\mathbf{q}", size=25, max_width=RIGHT_WIDTH - 0.65),
                mtex(r"s=c_q/h_q", size=32, color=GREEN, max_width=RIGHT_WIDTH - 0.65),
            ],
            buff=0.30,
        )
        put_in_panel(eqs, rp)

        bp = bottom_panel("Algorithm 2 branch logic")
        branches = VGroup(
            mtex(r"h_q=0:\;\mathrm{ignore}", size=20, color=GREY),
            mtex(r"c_q>0:\;s=c_q/h_q", size=20),
            mtex(r"c_q\leq0,\;h_q>0:\;s=0", size=20),
            mtex(r"\mathrm{else}:\;s=\infty", size=20, color=GREY),
        ).arrange(RIGHT, buff=0.38)
        put_in_panel(branches, bp, y_shift=0.05)

        self.play(FadeIn(hc), FadeIn(cc), Create(face), FadeIn(eqs), run_time=MED)
        for obj in eqs:
            self.play(Indicate(obj, color=BLUE, scale_factor=1.04), run_time=0.38)
        self.play(Create(current_ray), FadeIn(crossing), run_time=SLOW)
        self.play(FadeIn(bp), LaggedStartMap(FadeIn, branches, lag_ratio=0.18), run_time=SLOW)
        self.wait(PAUSE)
        self.clear_scene()

    # -------------------------------------------------------------------------
    # Scene 8
    # -------------------------------------------------------------------------

    def scene_verify(self):
        h, d, r, rp, sb = self.algorithm_base(
            "8. Verify the candidate",
            "A second nearest-site query checks whether this face is truly next.",
            4,
        )

        seg = self.segments[self.demo_k]
        nxt = self.segments[min(self.demo_k + 1, len(self.segments) - 1)]

        hc = self.highlight_cell(seg.index, YELLOW, opacity=0.32)
        cc = self.highlight_cell(nxt.index, RED, opacity=0.32)
        accepted = Line(data_to_manim(seg.start), data_to_manim(seg.end), color=BLUE, stroke_width=8)
        rcand = Dot(data_to_manim(seg.end + 0.015 * self.rhat), radius=0.09, color=BLUE)

        eqs = vstack(
            [
                mtex(r"\mathbf{r}_{\rm cand}=\mathbf{r}+dl_{\rm local}\hat{\mathbf{r}}", size=25, max_width=RIGHT_WIDTH - 0.65),
                mtex(r"I_{\rm cand}=\mathrm{tree\_nearest}(\mathbf{r}_{\rm cand},T)", size=24, max_width=RIGHT_WIDTH - 0.65),
                mtex(r"I_{\rm cand}\in\{I_{\rm cur},I_{\rm end}\}", size=26, color=GREEN, max_width=RIGHT_WIDTH - 0.65),
            ],
            buff=0.34,
        )
        put_in_panel(eqs, rp)

        bp = bottom_panel("Rejected candidate")
        fail = tex_text(
            "If another cell lies in front, store the failed candidate and continue.",
            size=20,
            color=ORANGE,
            max_width=BOTTOM_WIDTH - 0.8,
        )
        put_in_panel(fail, bp, y_shift=0.05)

        self.play(FadeIn(hc), FadeIn(cc), Create(accepted), FadeIn(rcand), FadeIn(eqs), run_time=MED)
        self.play(FadeIn(bp), FadeIn(fail), run_time=MED)
        self.wait(PAUSE)
        self.clear_scene()

    # -------------------------------------------------------------------------
    # Scene 9
    # -------------------------------------------------------------------------

    def scene_save(self):
        h, d, r, rp, sb = self.algorithm_base(
            "9. Save and advance",
            "Accepted crossings become rows in the ray file.",
            5,
        )

        seg = self.segments[self.demo_k]
        accepted = Line(data_to_manim(seg.start), data_to_manim(seg.end), color=BLUE, stroke_width=8)
        marker = Dot(data_to_manim(seg.start), radius=0.09, color=BLUE)

        eqs = vstack(
            [
                mtex(r"\mathrm{append}\;I_{\rm cur}\rightarrow\mathcal{I}", size=26, color=YELLOW, max_width=RIGHT_WIDTH - 0.65),
                mtex(r"\mathrm{append}\;dl_{\rm local}\rightarrow\Delta x", size=26, color=GREEN, max_width=RIGHT_WIDTH - 0.65),
                mtex(r"\mathbf{r}\leftarrow\mathbf{r}+dl_{\rm local}\hat{\mathbf{r}}", size=24, color=BLUE, max_width=RIGHT_WIDTH - 0.65),
                mtex(r"I_{\rm cur}\leftarrow I_{\rm end}", size=26, max_width=RIGHT_WIDTH - 0.65),
            ],
            buff=0.32,
        )
        put_in_panel(eqs, rp)

        bp = bottom_panel("One saved row")
        row = VGroup(
            tex_text("cell index:", size=19, color=GREY),
            tex_text(str(seg.index), size=19, color=YELLOW),
            tex_text("path length:", size=19, color=GREY),
            tex_text(f"{seg.length:.2f}", size=19, color=GREEN),
        ).arrange(RIGHT, buff=0.25)
        put_in_panel(row, bp, y_shift=0.05)

        self.play(FadeIn(eqs), Create(accepted), FadeIn(marker), run_time=MED)
        self.play(FadeIn(bp), FadeIn(row), run_time=MED)
        self.play(marker.animate.move_to(data_to_manim(seg.end)), run_time=SLOW)
        self.wait(PAUSE)
        self.clear_scene()

    # -------------------------------------------------------------------------
    # Scene 10
    # -------------------------------------------------------------------------

    def scene_repeat(self):
        h = heading(
            "10. Repeat until the final cell",
            "Only accepted cell IDs and path lengths are stored.",
        )
        d = self.diagram(cell_opacity=0.13)
        r = self.ray(color=GREY, width=2.6)
        marker = Dot(data_to_manim(self.r0), radius=0.09, color=BLUE)

        rp = panel(3.05, 3.35, [4.65, -0.20, 0], "Saved rows")
        header = VGroup(
            tex_text("j", size=15, color=GREY),
            tex_text("cell", size=15, color=YELLOW),
            tex_text("dx", size=15, color=GREEN),
        ).arrange(RIGHT, buff=0.38)
        header.next_to(rp[1], DOWN, buff=0.24)

        bp = bottom_panel("Progress")
        line_bg = Line(
            [-(BOTTOM_WIDTH - 1.0) / 2, BOTTOM_CENTER[1] - 0.05, 0],
            [(BOTTOM_WIDTH - 1.0) / 2, BOTTOM_CENTER[1] - 0.05, 0],
            color=GREY_DARK,
            stroke_width=5,
        )
        line_fg = Line(line_bg.get_start(), line_bg.get_start(), color=GREEN, stroke_width=5)

        self.play(FadeIn(h), FadeIn(d), FadeIn(r), FadeIn(marker), FadeIn(rp), FadeIn(header), FadeIn(bp), Create(line_bg), run_time=MED)

        rows = VGroup()
        path_lines = VGroup()
        travelled = 0.0

        for j, seg in enumerate(self.segments):
            hc = self.highlight_cell(seg.index, YELLOW, opacity=0.34)
            line = Line(data_to_manim(seg.start), data_to_manim(seg.end), color=BLUE, stroke_width=7)
            glow = Line(data_to_manim(seg.start), data_to_manim(seg.end), color=BLUE, stroke_width=13).set_opacity(0.16)

            travelled += seg.length
            frac = min(1.0, travelled / self.total_length)
            new_line = Line(
                line_bg.get_start(),
                line_bg.get_start() + RIGHT * line_bg.width * frac,
                color=GREEN,
                stroke_width=5,
            )

            self.play(FadeIn(hc), run_time=0.18)
            self.play(
                Create(glow),
                Create(line),
                marker.animate.move_to(data_to_manim(seg.end)),
                Transform(line_fg, new_line),
                run_time=0.65,
            )

            path_lines.add(glow, line)

            if j < 6:
                row = VGroup(
                    tex_text(str(j + 1), size=15),
                    tex_text(str(seg.index), size=15, color=YELLOW),
                    tex_text(f"{seg.length:.2f}", size=15, color=GREEN),
                ).arrange(RIGHT, buff=0.42)
                row.next_to(header, DOWN, buff=0.20 + 0.28 * j)
                rows.add(row)
                self.play(FadeIn(row), FadeOut(hc), run_time=0.22)
            else:
                self.play(FadeOut(hc), run_time=0.12)

        self.play(FadeOut(VGroup(rp, header, rows, bp, line_bg, line_fg)), run_time=FAST)

        final = bottom_panel("Final geometric ray")
        ids = ",".join(str(s.index) for s in self.segments[:7])
        dxs = ",".join(f"{s.length:.2f}" for s in self.segments[:7])
        if len(self.segments) > 7:
            ids += r",\ldots"
            dxs += r",\ldots"

        arrays = vstack(
            [
                mtex(r"\mathcal{I}=\{" + ids + r"\}", size=23, color=YELLOW, max_width=BOTTOM_WIDTH - 0.8),
                mtex(r"\Delta x=\{" + dxs + r"\}", size=23, color=GREEN, max_width=BOTTOM_WIDTH - 0.8),
            ],
            buff=0.04,
        )
        put_in_panel(arrays, final, y_shift=0.02)

        self.play(FadeIn(final), Write(arrays), run_time=SLOW)
        self.wait(PAUSE)
        self.clear_scene()

    # -------------------------------------------------------------------------
    # Scene 11
    # -------------------------------------------------------------------------

    def scene_physics(self):
        h = heading(
            "11. Geometry becomes physics",
            "A crossed cell contributes column density over its path length.",
        )
        d = self.diagram(cell_opacity=0.10)
        r = self.ray(color=GREY, width=2.5)

        lines = VGroup()
        for seg in self.segments:
            lines.add(Line(data_to_manim(seg.start), data_to_manim(seg.end), color=BLUE, stroke_width=6))

        rp = right_panel("Column contribution")
        eqs = vstack(
            [
                mtex(r"\Delta N_{{\rm ion},j}=n_{{\rm ion},j}\Delta x_j", size=31, color=GREEN, max_width=RIGHT_WIDTH - 0.65),
                mtex(r"N_{\rm ion}=\sum_j n_{{\rm ion},j}\Delta x_j", size=31, color=YELLOW, max_width=RIGHT_WIDTH - 0.65),
            ],
            buff=0.55,
        )
        put_in_panel(eqs, rp)

        tags = VGroup()
        for i, seg in enumerate(self.segments[:4]):
            mid = 0.5 * (seg.start + seg.end)
            tag = label_box(tex_text(f"cell {seg.index}: n, T, v", size=13), opacity=0.88)
            tag.move_to(data_to_manim(mid) + UP * (0.20 + 0.05 * (i % 2)))
            tags.add(tag)

        self.play(FadeIn(h), FadeIn(d), FadeIn(r), run_time=MED)
        self.play(Create(lines), run_time=SLOW)
        self.play(FadeIn(tags, lag_ratio=0.10), run_time=MED)
        self.play(FadeIn(rp), Write(eqs), run_time=SLOW)
        self.wait(PAUSE)
        self.clear_scene()

    # -------------------------------------------------------------------------
    # Scene 12
    # -------------------------------------------------------------------------

    def scene_spectrum(self):
        h = heading(
            "12. From ray file to spectrum",
            "Velocity shifts place absorption components in wavelength space.",
        )

        top = panel(11.70, 1.05, [0.0, 2.28, 0], "Spectral bookkeeping")
        eqs = VGroup(
            mtex(r"z_{\rm dop}=v_{\rm los}/c", size=23, color=BLUE),
            mtex(r"z_{\rm eff}=(1+z_{\rm cosmo})(1+z_{\rm dop})-1", size=23, color=YELLOW),
            mtex(r"F=e^{-\tau}", size=27, color=GREEN),
        ).arrange(RIGHT, buff=0.58)
        put_in_panel(eqs, top, y_shift=0.04)

        axes = Axes(
            x_range=[-500, 500, 250],
            y_range=[0, 1.12, 0.5],
            x_length=10.5,
            y_length=3.10,
            tips=False,
            axis_config={"color": PANEL_EDGE, "stroke_width": 2, "include_ticks": True},
        ).move_to([0.0, -0.90, 0])

        xlabel = tex_text("velocity [km/s]", size=18, color=GREY)
        xlabel.next_to(axes, DOWN, buff=0.18)
        ylabel = tex_text("normalized flux", size=18, color=GREY).rotate(PI / 2)
        ylabel.next_to(axes, LEFT, buff=0.16)

        xvals = np.linspace(-500, 500, 750)
        comps = [(-190, 0.42, 55), (15, 0.70, 42), (205, 0.30, 74)]

        def curve_for(n: int, color: str):
            tau = np.zeros_like(xvals)
            for center, depth, width in comps[:n]:
                tau0 = -math.log(max(1e-3, 1.0 - depth))
                tau += tau0 * np.exp(-0.5 * ((xvals - center) / width) ** 2)
            flux = np.exp(-tau)
            curve = VMobject(color=color, stroke_width=4)
            curve.set_points_as_corners([axes.c2p(float(x), float(y)) for x, y in zip(xvals, flux)])
            return curve

        self.play(FadeIn(h), FadeIn(top), Write(eqs), run_time=MED)
        self.play(Create(axes), FadeIn(xlabel), FadeIn(ylabel), run_time=MED)

        c = curve_for(1, BLUE)
        self.play(Create(c), run_time=SLOW)
        self.wait(0.30)
        self.play(Transform(c, curve_for(2, PURPLE)), run_time=SLOW)
        self.wait(0.30)
        self.play(Transform(c, curve_for(3, BLUE)), run_time=SLOW)
        self.wait(PAUSE)
        self.clear_scene()

    # -------------------------------------------------------------------------
    # Scene 13
    # -------------------------------------------------------------------------

    def scene_compare(self):
        h = heading(
            "13. Relation to yt / Trident LightRay",
            "The spectrum engine can be reused, but the ray geometry is different.",
        )

        left = panel(5.35, 3.15, [-3.18, 0.05, 0], "yt / Trident LightRay")
        lt = vstack(
            [
                tex_text("general yt ray sampling", size=21),
                tex_text("Arepo can be SPH-like", size=21),
                tex_text("smoothing-length contribution", size=21),
                tex_text("not expected to match exactly", size=20, color=GREY),
            ],
            buff=0.23,
        )
        put_in_panel(lt, left, y_shift=0.05)

        right = panel(5.35, 3.15, [3.18, 0.05, 0], "SALSA-style meshless")
        rt = vstack(
            [
                tex_text("Voronoi generating sites", size=21),
                tex_text("geometric path lengths", size=21, color=GREEN),
                tex_text("no full mesh stored", size=21),
                tex_text("compatible ray file", size=21),
            ],
            buff=0.23,
        )
        put_in_panel(rt, right, y_shift=0.05)

        bp = bottom_panel()
        msg = tex_text(
            "Expected comparison: same spectrum machinery, different ray-sampling assumptions.",
            size=21,
            color=YELLOW,
            max_width=BOTTOM_WIDTH - 0.8,
        )
        put_in_panel(msg, bp, y_shift=0.05)

        self.play(FadeIn(h), run_time=FAST)
        self.play(FadeIn(left), FadeIn(lt), run_time=MED)
        self.play(FadeIn(right), FadeIn(rt), run_time=MED)
        self.play(FadeIn(bp), FadeIn(msg), run_time=MED)
        self.wait(PAUSE)
        self.clear_scene()

    # -------------------------------------------------------------------------
    # Scene 14
    # -------------------------------------------------------------------------

    def scene_summary(self):
        h = heading(
            "Summary",
            "Nearest queries plus face intersections produce the ray file.",
        )

        card = panel(11.25, 4.20, [0.0, -0.20, 0])
        bullets = vstack(
            [
                tex_text("1. nearest tree identifies the current cell", size=27, max_width=10.2),
                tex_text("2. bisection and face intersection find the next crossing", size=27, max_width=10.2),
                tex_text("3. ordered cell IDs and path lengths feed spectra", size=27, max_width=10.2),
            ],
            buff=0.36,
            aligned_edge=LEFT,
        )
        bullets.move_to(card[0].get_center() + UP * 0.55)

        flow = mtex(
            r"\{\mathbf{x}_i\},\mathbf{r}_0,\hat{\mathbf{r}},S"
            r"\;\rightarrow\;(\mathcal{I},\Delta x)"
            r"\;\rightarrow\;F(\lambda)",
            size=30,
            color=YELLOW,
            max_width=10.4,
        )
        flow.next_to(bullets, DOWN, buff=0.58)

        self.play(FadeIn(h), FadeIn(card), run_time=MED)
        self.play(LaggedStartMap(FadeIn, bullets, lag_ratio=0.18), run_time=1.25)
        self.play(Write(flow), run_time=SLOW)
        self.wait(1.60)
        self.clear_scene()