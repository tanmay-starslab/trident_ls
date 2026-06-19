"""
Quasar -> synthetic spectrum, hybrid edition.

Structure follows the original short version (Title-less; QSO+galaxy ->
sightline -> flatten -> absorption dips -> total flux -> LSF + noise ->
final fit table), but the cosmic and sightline scenes use the SOTU 3D
objects: galaxy group with sphere halo, IGrM particle cloud, spiral
galaxies, 3D quasar, and a single continuous 3D sightline. Sphere is
rotated quickly. Heavier noise on the observed spectrum. Voigt-fit
observables table from the SOTU build is appended at the end.

No titles, no chapter labels, no taglines — those are added externally.

Render:
    /Users/wavefunction/github_repos/trident_ls/salsa/bin/python -m manim render -ql \
        --disable_caching dev/meshless_tests/quasar_to_spectrum_animation.py QuasarToSpectrum
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import wofz
from scipy.optimize import curve_fit

from manim import *


# =============================================================================
# Render configuration
# =============================================================================

config.pixel_width = 1920
config.pixel_height = 1080
config.frame_rate = 60
config.background_color = "#000000"
config.renderer = "cairo"

# Density of 3D objects — kept modest so cairo can render in reasonable time
N_IGRM        = 70
N_GALAXY_ARM  = 14
N_QSO_SPIKES  = 8
HALO_RES      = (8, 16)
CLOUD_RES     = (5, 10)
SPECTRUM_N    = 700


# =============================================================================
# Palette (carried over from the SOTU + meshless style)
# =============================================================================

COL_BG          = "#000000"
COL_PANEL       = "#0B1020"
COL_PANEL_SOFT  = "#0E1530"
COL_PANEL_STROKE = "#2B334A"

COL_TEXT        = "#EAEAF2"
COL_TEXT_DIM    = "#C5CCDA"
COL_MUTED       = "#9AA4B2"
COL_FAINT       = "#5C6477"

COL_QSO         = "#FFD166"
COL_QSO_BEAM    = "#FFE08A"
COL_OBSERVER    = "#EAEAF2"

COL_HALO        = "#6366F1"
COL_VIRIAL      = "#A5B4FC"

COL_OVI         = "#38BDF8"
COL_WARM        = "#FACC15"
COL_HOT         = "#FB7185"
COL_COOL        = "#34D399"
COL_UVB         = "#A78BFA"

COL_TRUE        = "#38BDF8"
COL_LSF         = "#F5C542"
COL_NOISY       = "#FF8E72"
COL_FIT         = "#7CFFB2"
COL_RESID       = "#C0C8D6"

COL_PURPLE      = "#C084FC"
COL_GREEN       = "#63E6BE"
COL_GOLD        = "#F5C542"


# =============================================================================
# Physical constants
# =============================================================================

C_KMS         = 299792.458
OVI_LAMBDA_1  = 1031.926


# =============================================================================
# Absorber components (mirror SOTU)
# =============================================================================

@dataclass
class AbsorberComponent:
    velocity_kms: float
    logN: float
    b_kms: float
    color: str
    label: str


CLOUDS = [
    AbsorberComponent(-255, 13.45, 24, COL_PURPLE, "C1"),
    AbsorberComponent(-115, 13.75, 32, COL_OVI,    "C2"),
    AbsorberComponent(  35, 14.35, 44, COL_GOLD,   "C3"),
    AbsorberComponent( 160, 13.92, 28, COL_HOT,    "C4"),
    AbsorberComponent( 265, 13.58, 22, COL_GREEN,  "C5"),
]


# =============================================================================
# Spectrum physics
# =============================================================================

def voigt_tau(v, components, strength_scale=0.48):
    tau = np.zeros_like(v, dtype=float)
    for c in components:
        amp = strength_scale * 10 ** (c.logN - 14.0)
        sigma = c.b_kms / np.sqrt(2.0)
        gamma = 0.075 * c.b_kms
        z = ((v - c.velocity_kms) + 1j * gamma) / (sigma * np.sqrt(2))
        profile = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
        if np.max(profile) > 0:
            profile /= np.max(profile)
        tau += amp * profile
    return tau


def voigt_sum_model(v, *params):
    n = len(params) // 3
    tau = np.zeros_like(v)
    for i in range(n):
        v0, lN, b = params[3 * i:3 * i + 3]
        amp = 0.48 * 10 ** (lN - 14.0)
        sigma = b / np.sqrt(2.0)
        gamma = 0.075 * b
        z = ((v - v0) + 1j * gamma) / (sigma * np.sqrt(2))
        profile = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
        if np.max(profile) > 0:
            profile /= np.max(profile)
        tau += amp * profile
    return np.exp(-tau)


def cos_g130m_lsf_kernel(v, R=18000.0):
    fwhm = C_KMS / R
    sc = fwhm / 2.354820045
    sm = 2.8 * sc
    sw = 7.0 * sc
    gc = np.exp(-0.5 * (v / sc) ** 2) / (sc * np.sqrt(2 * np.pi))
    gm = np.exp(-0.5 * (v / sm) ** 2) / (sm * np.sqrt(2 * np.pi))
    gw = np.exp(-0.5 * (v / sw) ** 2) / (sw * np.sqrt(2 * np.pi))
    k = 0.62 * gc + 0.26 * gm + 0.12 * gw
    return k / k.sum()


def _load_lsf():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(here, "avg_COS_G130M.txt"), "data/avg_COS_G130M.txt"]:
        try:
            d = np.loadtxt(p)
            if d.ndim == 2 and d.shape[1] >= 2:
                x, k = d[:, 0], d[:, 1]
            else:
                x = np.arange(len(d)) - len(d) // 2
                k = d.ravel()
            k = np.where(np.isfinite(k), k, 0.0)
            k = np.maximum(k, 0.0)
            if k.sum() > 0:
                k = k / k.sum()
            return x, k
        except Exception:
            pass
    return None, None


_LSF_PIX, _LSF_K = _load_lsf()
_PIX_TO_KMS = 9.97e-3 / 1032.0 * C_KMS
USING_TABULATED_LSF = _LSF_PIX is not None


def convolve_with_lsf(flux, v, R=18000.0):
    dv = float(np.median(np.diff(v)))
    if USING_TABULATED_LSF:
        x_kms = _LSF_PIX * _PIX_TO_KMS * 1.20
        v_kern = np.arange(x_kms.min(), x_kms.max() + dv, dv)
        k = np.interp(v_kern, x_kms, _LSF_K, left=0.0, right=0.0)
        k = np.maximum(k, 0.0)
        if k.sum() > 0:
            k = k / k.sum()
    else:
        kv = np.arange(-360.0, 360.0 + dv, dv)
        k = cos_g130m_lsf_kernel(kv, R=R)
    pad = len(k) // 2
    padded = np.pad(flux, pad_width=pad, mode="edge")
    conv = fftconvolve(padded, k, mode="same")
    return conv[pad:pad + len(flux)]


def add_noise(flux, snr=6.0, seed=42):
    rng = np.random.default_rng(seed)
    sigma = 1.0 / snr
    return np.clip(flux + rng.normal(0.0, sigma, size=len(flux)), -0.10, 1.40)


def bin_spectrum(x, y, factor=3):
    n = (len(x) // factor) * factor
    xb = x[:n].reshape(-1, factor).mean(axis=1)
    yb = y[:n].reshape(-1, factor).mean(axis=1)
    return xb, yb


def fit_voigt(velocity, flux, truth):
    p0, lo, hi = [], [], []
    for c in truth:
        p0 += [c.velocity_kms, c.logN, c.b_kms]
        lo += [c.velocity_kms - 60.0, 12.0, 5.0]
        hi += [c.velocity_kms + 60.0, 15.5, 80.0]
    try:
        popt, _ = curve_fit(voigt_sum_model, velocity, flux,
                            p0=p0, bounds=(lo, hi), maxfev=5000)
    except Exception:
        popt = np.array(p0)
    return popt


# =============================================================================
# 3D object builders (from SOTU)
# =============================================================================

def random_points_in_ellipsoid(n, scales, center, seed):
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=(n, 3))
    vec /= np.linalg.norm(vec, axis=1)[:, None]
    rad = rng.uniform(0, 1, n) ** (1 / 3)
    pts = vec * rad[:, None]
    pts[:, 0] *= scales[0]
    pts[:, 1] *= scales[1]
    pts[:, 2] *= scales[2]
    pts += np.array(center)[None, :]
    return pts


def make_spiral_galaxy_3d(center, radius=0.35, color=COL_TEXT, seed=0,
                          inclination_deg=18.0, n_arms=2):
    rng = np.random.default_rng(seed)
    g = Group()

    core = Dot3D(point=center, radius=radius * 0.09, color=color)
    core.set_opacity(1.0)
    g.add(core)

    for i in range(4):
        haze = Dot3D(point=center, radius=radius * (0.13 + 0.04 * i), color=color)
        haze.set_opacity(0.18 - 0.035 * i)
        g.add(haze)

    inc = np.deg2rad(inclination_deg)
    rot_x = np.array([
        [1, 0, 0],
        [0, np.cos(inc), -np.sin(inc)],
        [0, np.sin(inc), np.cos(inc)],
    ])

    for arm in range(n_arms):
        phase = 2 * np.pi * arm / n_arms
        for i in range(N_GALAXY_ARM):
            t = i / max(N_GALAXY_ARM - 1, 1)
            theta = 5.8 * t + phase
            r = radius * (0.13 + 1.05 * t)
            p = np.array([
                r * np.cos(theta),
                0.48 * r * np.sin(theta),
                rng.normal(0.0, radius * 0.018),
            ])
            p = rot_x @ p
            tint = "#A8C7FF" if t > 0.55 and rng.random() < 0.35 else color
            d = Dot3D(point=center + p, radius=rng.uniform(0.010, 0.025), color=tint)
            d.set_opacity(rng.uniform(0.45, 0.95))
            g.add(d)

    edge = Circle(radius=radius * 1.10)
    edge.set_stroke(color=color, width=1.0, opacity=0.22)
    edge.set_fill(color=color, opacity=0.035)
    edge.move_to(center)
    edge.rotate(np.deg2rad(inclination_deg), axis=RIGHT)
    g.add(edge)
    return g


def make_igrm_cloud_3d(n=N_IGRM, center=(0, 0, 0),
                       scales=(2.3, 1.6, 1.2), seed=0):
    rng = np.random.default_rng(seed)
    pts = random_points_in_ellipsoid(n=n, scales=scales, center=center, seed=seed)
    cloud = Group()
    for p in pts:
        rr = np.linalg.norm((p - np.array(center)) / np.array(scales))
        roll = rng.uniform()
        if rr < 0.45 and roll < 0.6:
            col = COL_OVI
        elif rr < 0.7 and roll < 0.55:
            col = COL_WARM
        elif roll < 0.65:
            col = COL_HOT
        else:
            col = COL_COOL
        dot = Dot3D(
            point=p,
            radius=rng.uniform(0.012, 0.034) * (1.15 - 0.35 * rr),
            color=col,
        )
        dot.set_opacity(rng.uniform(0.10, 0.34) * (1.2 - 0.5 * rr))
        cloud.add(dot)
    return cloud


def make_quasar_3d(position, color=COL_QSO):
    pos = np.array(position)
    qso = Group()

    core = Dot3D(point=pos, radius=0.14, color=color)
    core.set_opacity(1.0)
    qso.add(core)

    for r, op in [(0.20, 0.35), (0.28, 0.18), (0.36, 0.08)]:
        glow = Dot3D(point=pos, radius=r, color=color)
        glow.set_opacity(op)
        qso.add(glow)

    for disk_r, disk_op in [(0.42, 0.60), (0.22, 0.45)]:
        disk = Circle(radius=disk_r)
        disk.set_fill(color, opacity=0.10)
        disk.set_stroke(color, width=2.0, opacity=disk_op)
        disk.move_to(pos)
        disk.rotate(65 * DEGREES, axis=RIGHT, about_point=pos)
        disk.rotate(25 * DEGREES, axis=OUT, about_point=pos)
        qso.add(disk)

    rng = np.random.default_rng(11)
    for _ in range(N_QSO_SPIKES):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        start = pos + 0.18 * d
        end = pos + rng.uniform(0.32, 0.60) * d
        ray = Line3D(start=start, end=end, color=color, thickness=0.010)
        qso.add(ray)
    return qso


def make_irregular_gas_cell(center, color=COL_OVI, radius=0.22, n_lobes=4, seed=0):
    rng = np.random.default_rng(seed)
    ctr = np.array(center)
    g = Group()

    main = Sphere(radius=radius * 0.72, resolution=CLOUD_RES)
    main.set_fill(color, opacity=0.34)
    main.set_stroke(color, width=0.5, opacity=0.75)
    main.move_to(ctr)
    g.add(main)

    for _ in range(n_lobes):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        off = rng.uniform(0.09, 0.19) * d
        lobe = Sphere(radius=rng.uniform(0.07, 0.14) * (radius / 0.22),
                      resolution=CLOUD_RES)
        lobe.set_fill(color, opacity=rng.uniform(0.20, 0.38))
        lobe.set_stroke(color, width=0.3, opacity=0.50)
        lobe.move_to(ctr + off)
        g.add(lobe)
    return g


def make_uv_background_field(scene_radius=3.25, n_arrows=10,
                             color=COL_UVB, opacity=0.55):
    arrows = Group()
    for i in range(n_arrows):
        a = 2 * np.pi * i / n_arrows
        start = scene_radius * np.array([np.cos(a), np.sin(a), 0.0])
        end = 0.60 * start
        arr = Arrow(start=start, end=end, buff=0,
                    stroke_width=2.6, max_tip_length_to_length_ratio=0.28)
        arr.set_color(color)
        arr.set_opacity(opacity)
        arrows.add(arr)
    return arrows


# =============================================================================
# Generic plot/panel helpers
# =============================================================================

def make_panel(width, height, fill=COL_PANEL, stroke=COL_PANEL_STROKE,
               opacity=0.96, corner_radius=0.18,
               stroke_opacity=0.90, stroke_width=2.0):
    p = RoundedRectangle(width=width, height=height, corner_radius=corner_radius)
    p.set_fill(fill, opacity=opacity)
    p.set_stroke(stroke, width=stroke_width, opacity=stroke_opacity)
    return p


def make_curve_from_arrays(ax, x, y, color=COL_TEXT, width=4.0, opacity=1.0):
    pts = np.array([ax.c2p(float(xx), float(yy)) for xx, yy in zip(x, y)])
    c = VMobject()
    c.set_points_as_corners(pts)
    c.set_stroke(color=color, width=width, opacity=opacity)
    return c


def make_step_curve_from_arrays(ax, x, y, color=COL_NOISY, width=2.2, opacity=1.0):
    n = len(x)
    dx = np.median(np.diff(x))
    edges = np.empty(n + 1)
    edges[0] = x[0] - dx / 2
    edges[1:-1] = (x[:-1] + x[1:]) / 2
    edges[-1] = x[-1] + dx / 2
    pts = []
    for i in range(n):
        pts.append(ax.c2p(float(edges[i]),     float(y[i])))
        pts.append(ax.c2p(float(edges[i + 1]), float(y[i])))
        if i < n - 1:
            pts.append(ax.c2p(float(edges[i + 1]), float(y[i + 1])))
    c = VMobject()
    c.set_points_as_corners(pts)
    c.set_stroke(color=color, width=width, opacity=opacity)
    return c


# =============================================================================
# Main scene
# =============================================================================

class QuasarToSpectrum(ThreeDScene):

    def construct(self):
        self.camera.background_color = COL_BG
        self._hud_set = []

        self.scene_cosmic_3d()
        self.scene_continuous_sightline()
        self.scene_flatten_to_1d()
        self.scene_absorption_dips()
        self.scene_total_flux()
        self.scene_lsf_and_noise()
        self.scene_fit_table()

    # ------------------------------------------------------------------
    # HUD helpers
    # ------------------------------------------------------------------
    def add_hud(self, *mobs):
        for m in mobs:
            self.add_fixed_in_frame_mobjects(m)
            if m not in self._hud_set:
                self._hud_set.append(m)

    def remove_hud(self, *mobs):
        self.remove(*mobs)
        for m in mobs:
            if m in self._hud_set:
                self._hud_set.remove(m)

    # ------------------------------------------------------------------
    # Scene A — 3D cosmic scene (galaxy group + halo + IGrM + QSO)
    # ------------------------------------------------------------------
    def scene_cosmic_3d(self):
        self.set_camera_orientation(phi=62 * DEGREES, theta=-52 * DEGREES)

        qso_pos = np.array([-4.8, 2.4, 2.0])
        observer_pos = np.array([4.8, -2.0, -1.5])
        group_center = np.array([0.0, 0.0, 0.0])

        halo_shell = Sphere(radius=2.40, resolution=HALO_RES)
        halo_shell.set_fill(COL_HALO, opacity=0.05)
        halo_shell.set_stroke(COL_HALO, width=0.35, opacity=0.22)

        virial_ring = Circle(radius=2.40)
        virial_ring.set_stroke(COL_VIRIAL, width=2.1, opacity=0.70)
        virial_ring.rotate(PI / 2, axis=RIGHT)

        igrm = make_igrm_cloud_3d(n=N_IGRM, center=group_center,
                                  scales=(2.20, 1.50, 1.20), seed=91)

        galaxies = Group(
            make_spiral_galaxy_3d(np.array([-0.65,  0.25,  0.18]), radius=0.36, seed=1),
            make_spiral_galaxy_3d(np.array([ 0.78, -0.32, -0.10]), radius=0.28, seed=2),
            make_spiral_galaxy_3d(np.array([ 0.15,  0.78,  0.05]), radius=0.24, seed=3),
            make_spiral_galaxy_3d(np.array([-0.95, -0.72,  0.12]), radius=0.22, seed=4),
            make_spiral_galaxy_3d(np.array([ 0.50,  0.10,  0.30]), radius=0.18, seed=5),
        )

        qso = make_quasar_3d(qso_pos)
        uvb = make_uv_background_field()

        self.add_hud(uvb)
        self.play(FadeIn(uvb, run_time=0.7))
        self.play(FadeIn(qso, run_time=0.9))
        self.play(
            FadeIn(halo_shell, run_time=1.1),
            Create(virial_ring, run_time=1.1),
            LaggedStart(*[FadeIn(g, scale=0.45) for g in galaxies],
                        lag_ratio=0.10, run_time=1.3),
        )
        self.play(FadeIn(igrm, run_time=0.9))

        # Fast camera rotation — sweep around theta
        self.begin_ambient_camera_rotation(rate=0.65, about="theta")
        self.wait(4.5)
        self.stop_ambient_camera_rotation()

        # Return to a clean canonical view for the sightline shot
        self.move_camera(phi=62 * DEGREES, theta=-52 * DEGREES, run_time=0.9,
                         rate_func=rate_functions.ease_in_out_cubic)

        self._cosmic = dict(
            qso_pos=qso_pos,
            observer_pos=observer_pos,
            group_center=group_center,
            qso=qso, halo_shell=halo_shell, virial_ring=virial_ring,
            igrm=igrm, galaxies=galaxies, uvb=uvb,
        )

    # ------------------------------------------------------------------
    # Scene B — continuous sightline through the gas
    # ------------------------------------------------------------------
    def scene_continuous_sightline(self):
        c = self._cosmic
        qso_pos = c["qso_pos"]
        observer_pos = c["observer_pos"]
        los_vec = observer_pos - qso_pos

        def P(t):
            return qso_pos + t * los_vec

        # ONE continuous sightline drawn in a single smooth Create
        sightline = Line3D(start=P(0.04), end=P(0.96),
                           color=COL_QSO_BEAM, thickness=0.034)
        self.play(Create(sightline, run_time=2.6,
                         rate_func=rate_functions.ease_in_out_sine))

        # 3D gas cells along the line, in colour order matching the absorbers
        t_cells = [0.37, 0.46, 0.55, 0.63, 0.72]
        ray_clouds = Group()
        for t, comp in zip(t_cells, CLOUDS):
            ray_clouds.add(make_irregular_gas_cell(
                center=P(t), color=comp.color, radius=0.24, n_lobes=4,
                seed=abs(hash(comp.label)) % (2 ** 31),
            ))

        self.play(LaggedStart(*[FadeIn(cl, scale=0.45) for cl in ray_clouds],
                              lag_ratio=0.10, run_time=1.4))
        self.wait(0.9)

        # Gentle continued rotation while the user takes in the sightline
        self.begin_ambient_camera_rotation(rate=0.50, about="theta")
        self.wait(2.4)
        self.stop_ambient_camera_rotation()

        self._sightline = dict(sightline=sightline, ray_clouds=ray_clouds)

    # ------------------------------------------------------------------
    # Scene C — flatten the 3D ray to an ordered 1D track
    # ------------------------------------------------------------------
    def scene_flatten_to_1d(self):
        c = self._cosmic
        s = self._sightline

        self.play(FadeOut(c["uvb"], run_time=0.4))
        self.remove_hud(c["uvb"])

        # Tilt camera to a near-top-down view
        self.move_camera(phi=10 * DEGREES, theta=-90 * DEGREES, run_time=1.4,
                         rate_func=rate_functions.ease_in_out_cubic)

        # Dim the 3D background as the ray flattens
        self.play(
            c["halo_shell"].animate.set_opacity(0.020),
            c["igrm"].animate.set_opacity(0.18),
            c["galaxies"].animate.set_opacity(0.22),
            c["qso"].animate.set_opacity(0.22),
            c["virial_ring"].animate.set_opacity(0.15),
            run_time=0.8,
        )

        # 1D track (HUD) — ordered colored cells
        panel_w = 11.4
        panel_h = 1.80
        ray_panel = make_panel(width=panel_w, height=panel_h, fill="#07101E")
        ray_panel.move_to(np.array([0.0, -0.10, 0.0]))

        ax_left = ray_panel.get_left()[0] + 0.90
        ax_right = ray_panel.get_right()[0] - 0.90
        ax_y = ray_panel.get_y()

        ray_axis = Line(np.array([ax_left, ax_y, 0.0]),
                        np.array([ax_right, ax_y, 0.0]))
        ray_axis.set_stroke(COL_QSO, width=4.4, opacity=0.92)

        xs = np.linspace(ax_left + 0.55, ax_right - 0.55, len(CLOUDS))
        cell_dots = Group()
        for x, comp in zip(xs, CLOUDS):
            cell = Circle(radius=0.20)
            cell.set_fill(comp.color, opacity=0.62)
            cell.set_stroke(comp.color, width=2.0, opacity=0.95)
            cell.move_to(np.array([x, ax_y, 0.0]))
            cell_dots.add(cell)

        self.add_hud(ray_panel, ray_axis)
        self.play(FadeIn(ray_panel), Create(ray_axis, run_time=0.7))

        self.add_hud(cell_dots)
        self.play(LaggedStart(*[FadeIn(c, scale=0.35) for c in cell_dots],
                              lag_ratio=0.10, run_time=1.0))
        self.wait(1.0)

        # Remove the 3D world entirely; the 2D/HUD work begins
        world = Group(
            c["halo_shell"], c["virial_ring"], c["igrm"], c["galaxies"],
            c["qso"], s["sightline"], s["ray_clouds"],
        )
        self.play(FadeOut(world, run_time=0.7),
                  FadeOut(ray_panel, run_time=0.7),
                  FadeOut(ray_axis, run_time=0.7))
        self.remove_hud(ray_panel, ray_axis)
        self.remove(*world)

        # Move camera flat (phi=0) so HUD axes sit naturally on screen
        self.move_camera(phi=0 * DEGREES, theta=-90 * DEGREES, run_time=0.55)

        self._track_cells = cell_dots

    # ------------------------------------------------------------------
    # Scene D — flux axes; each cell carves its own dip
    # ------------------------------------------------------------------
    def scene_absorption_dips(self):
        self._velocity = np.linspace(-500, 500, SPECTRUM_N)

        ax = Axes(
            x_range=[-500, 500, 250],
            y_range=[-0.05, 1.30, 0.25],
            x_length=12.0, y_length=3.2,
            tips=False,
            axis_config=dict(stroke_color=COL_MUTED, stroke_width=1.8,
                             include_numbers=False),
        )
        ax.move_to(np.array([0.0, -0.4, 0.0]))

        x_lab = MathTex(r"v\,[\mathrm{km/s}]", font_size=22, color=COL_MUTED)
        x_lab.next_to(ax, DOWN, buff=0.15)
        y_lab = MathTex(r"F/F_c", font_size=22, color=COL_MUTED)
        y_lab.next_to(ax, LEFT, buff=0.20)

        continuum = make_curve_from_arrays(ax, self._velocity,
                                           np.ones_like(self._velocity),
                                           color=COL_MUTED, width=1.6, opacity=0.55)

        # Move the small track dots up out of the spectrum region
        self.play(self._track_cells.animate.shift(UP * 2.5).scale(0.85), run_time=0.6)

        self.add_hud(ax, x_lab, y_lab, continuum)
        self.play(Create(ax), FadeIn(x_lab), FadeIn(y_lab), Create(continuum),
                  run_time=0.9)

        single_curves = []
        for i, comp in enumerate(CLOUDS):
            tau_i = voigt_tau(self._velocity, [comp], strength_scale=0.48)
            flux_i = np.exp(-tau_i)
            curve = make_curve_from_arrays(ax, self._velocity, flux_i,
                                           color=comp.color, width=2.6, opacity=0.92)
            ring = Circle(radius=0.27, color=comp.color, stroke_width=3.0)
            ring.set_fill(opacity=0)
            ring.move_to(self._track_cells[i].get_center())

            self.add_hud(ring)
            self.play(Create(ring, run_time=0.20))
            self.add_hud(curve)
            self.play(Create(curve, run_time=0.55))
            self.play(FadeOut(ring, run_time=0.15))
            self.remove_hud(ring)
            single_curves.append(curve)
        self.wait(0.5)

        self._ax = ax
        self._ax_labels = Group(x_lab, y_lab)
        self._continuum = continuum
        self._single_curves = single_curves

    # ------------------------------------------------------------------
    # Scene E — combine to total transmitted flux F = exp(-tau)
    # ------------------------------------------------------------------
    def scene_total_flux(self):
        flux_true = np.exp(-voigt_tau(self._velocity, CLOUDS, strength_scale=0.48))
        true_curve = make_curve_from_arrays(self._ax, self._velocity, flux_true,
                                            color=COL_TRUE, width=4.0)

        self.add_hud(true_curve)
        # Morph all the single-cloud curves into the joint flux
        self.play(*[Transform(c, true_curve.copy()) for c in self._single_curves],
                  run_time=1.1)
        self.play(*[FadeOut(c) for c in self._single_curves],
                  FadeIn(true_curve), run_time=0.40)
        for c in self._single_curves:
            self.remove_hud(c)

        flux_eq = MathTex(r"F_{\rm true}(v)=\exp[-\tau(v)]",
                          font_size=30, color=COL_TRUE)
        flux_eq.move_to(np.array([4.2, 2.55, 0.0]))
        self.add_hud(flux_eq)
        self.play(FadeIn(flux_eq, shift=UP * 0.10), run_time=0.6)
        self.wait(1.0)

        self._true_curve = true_curve
        self._flux_true = flux_true
        self._flux_eq = flux_eq

    # ------------------------------------------------------------------
    # Scene F — LSF blur, then heavy noise
    # ------------------------------------------------------------------
    def scene_lsf_and_noise(self):
        flux_lsf = convolve_with_lsf(self._flux_true, self._velocity, R=18000.0)
        lsf_curve = make_curve_from_arrays(self._ax, self._velocity, flux_lsf,
                                           color=COL_LSF, width=3.4)

        self.add_hud(lsf_curve)
        self.play(Transform(self._true_curve, lsf_curve), run_time=1.00)
        self.wait(0.5)

        # Heavy noise — drop SNR to ~5
        vb, fb_clean = bin_spectrum(self._velocity, flux_lsf, factor=3)
        fb = add_noise(fb_clean, snr=5.0, seed=137)
        noisy_curve = make_step_curve_from_arrays(self._ax, vb, fb,
                                                  color=COL_NOISY, width=2.4)
        self.add_hud(noisy_curve)
        self.play(Create(noisy_curve, run_time=1.10))
        self.wait(1.2)

        self._lsf_curve = lsf_curve
        self._noisy_curve = noisy_curve
        self._vb = vb
        self._fb = fb

    # ------------------------------------------------------------------
    # Scene G — Voigt-fit observables table (from SOTU build)
    # ------------------------------------------------------------------
    def scene_fit_table(self):
        # Fit on the noisy binned data
        popt = fit_voigt(self._vb, self._fb, CLOUDS)
        fitted = []
        for i, c in enumerate(CLOUDS):
            v0, lN, b = popt[3 * i], popt[3 * i + 1], popt[3 * i + 2]
            fitted.append(AbsorberComponent(v0, lN, b, c.color, c.label))

        # Overlay the fit on top of the noisy spectrum
        dense_fit = voigt_sum_model(self._velocity, *popt)
        fit_curve = make_curve_from_arrays(self._ax, self._velocity, dense_fit,
                                           color=COL_FIT, width=3.0, opacity=0.95)
        self.add_hud(fit_curve)
        self.play(Create(fit_curve, run_time=1.1))
        self.wait(0.9)

        # Clear the spectrum area and present the observables sheet
        sweep = [self._ax, self._ax_labels, self._continuum,
                 self._true_curve, self._lsf_curve, self._noisy_curve,
                 fit_curve, self._flux_eq, self._track_cells]
        self.play(*[FadeOut(m, run_time=0.5) for m in sweep])
        for m in sweep:
            self.remove_hud(m)

        sheet_w = 12.6
        sheet_h = 5.10
        sheet = make_panel(width=sheet_w, height=sheet_h,
                           fill=COL_PANEL, stroke=COL_PANEL_STROKE)
        sheet.move_to(np.array([0.0, -0.05, 0.0]))
        self.add_hud(sheet)
        self.play(FadeIn(sheet, run_time=0.55))

        comp_x   = -4.80
        v_true_x, v_fit_x = -3.10, -1.90
        n_true_x, n_fit_x = -0.60,  0.60
        b_true_x, b_fit_x =  2.00,  3.10

        top_y = sheet.get_top()[1]
        gh_y = top_y - 0.50
        sub_y = gh_y - 0.45

        headers = Group(
            MathTex(r"\mathrm{component}", font_size=20, color=COL_GOLD).move_to(np.array([comp_x, gh_y, 0.0])),
            MathTex(r"v\,[\mathrm{km/s}]", font_size=20, color=COL_GOLD).move_to(np.array([(v_true_x + v_fit_x) / 2, gh_y, 0.0])),
            MathTex(r"\log N_{\rm O\,VI}", font_size=20, color=COL_GOLD).move_to(np.array([(n_true_x + n_fit_x) / 2, gh_y, 0.0])),
            MathTex(r"b\,[\mathrm{km/s}]", font_size=20, color=COL_GOLD).move_to(np.array([(b_true_x + b_fit_x) / 2, gh_y, 0.0])),
        )
        sub_headers = Group(
            MathTex(r"\mathrm{true}", font_size=17, color=COL_TEXT_DIM).move_to(np.array([v_true_x, sub_y, 0.0])),
            MathTex(r"\mathrm{fit}",  font_size=17, color=COL_FIT     ).move_to(np.array([v_fit_x,  sub_y, 0.0])),
            MathTex(r"\mathrm{true}", font_size=17, color=COL_TEXT_DIM).move_to(np.array([n_true_x, sub_y, 0.0])),
            MathTex(r"\mathrm{fit}",  font_size=17, color=COL_FIT     ).move_to(np.array([n_fit_x,  sub_y, 0.0])),
            MathTex(r"\mathrm{true}", font_size=17, color=COL_TEXT_DIM).move_to(np.array([b_true_x, sub_y, 0.0])),
            MathTex(r"\mathrm{fit}",  font_size=17, color=COL_FIT     ).move_to(np.array([b_fit_x,  sub_y, 0.0])),
        )
        header_rule = Line(np.array([comp_x - 0.45, sub_y - 0.24, 0.0]),
                           np.array([b_fit_x + 0.45, sub_y - 0.24, 0.0]))
        header_rule.set_stroke(COL_GOLD, width=1.0, opacity=0.75)

        self.add_hud(headers, sub_headers, header_rule)
        self.play(FadeIn(headers), FadeIn(sub_headers), Create(header_rule))

        row_gap = 0.56
        first_row_y = sub_y - 0.55
        rows = Group()
        for i, (tr, ft) in enumerate(zip(CLOUDS, fitted)):
            y = first_row_y - i * row_gap
            dot = Dot(radius=0.080, color=tr.color)
            dot.move_to(np.array([comp_x - 0.30, y, 0.0]))
            name = MathTex(tr.label, font_size=20, color=tr.color)
            name.next_to(dot, RIGHT, buff=0.12)
            row = Group(
                Group(dot, name),
                MathTex(f"{tr.velocity_kms:+.0f}", font_size=20, color=COL_TEXT).move_to(np.array([v_true_x, y, 0.0])),
                MathTex(f"{ft.velocity_kms:+.0f}", font_size=20, color=COL_FIT ).move_to(np.array([v_fit_x,  y, 0.0])),
                MathTex(f"{tr.logN:.2f}",          font_size=20, color=COL_TEXT).move_to(np.array([n_true_x, y, 0.0])),
                MathTex(f"{ft.logN:.2f}",          font_size=20, color=COL_FIT ).move_to(np.array([n_fit_x,  y, 0.0])),
                MathTex(f"{tr.b_kms:.0f}",         font_size=20, color=COL_TEXT).move_to(np.array([b_true_x, y, 0.0])),
                MathTex(f"{ft.b_kms:.0f}",         font_size=20, color=COL_FIT ).move_to(np.array([b_fit_x,  y, 0.0])),
            )
            rows.add(row)
            if i < len(CLOUDS) - 1:
                sep = Line(np.array([comp_x - 0.45, y - row_gap / 2, 0.0]),
                           np.array([b_fit_x + 0.45, y - row_gap / 2, 0.0]))
                sep.set_stroke(COL_FAINT, width=0.6, opacity=0.40)
                rows.add(sep)

        self.add_hud(rows)
        self.play(LaggedStart(*[FadeIn(r, shift=0.08 * RIGHT)
                                for r in rows if isinstance(r, Group)],
                              lag_ratio=0.09, run_time=1.5))

        N_tot_truth = np.sum([10 ** c.logN for c in CLOUDS])
        N_tot_fit = np.sum([10 ** c.logN for c in fitted])
        sum_y = first_row_y - len(CLOUDS) * row_gap + 0.05

        sum_t = MathTex(r"\log N_{\rm O\,VI}^{\rm truth}=" + f"{np.log10(N_tot_truth):.2f}",
                        font_size=24, color=COL_TEXT).move_to(np.array([-0.25, sum_y, 0.0]))
        sum_f = MathTex(r"\log N_{\rm O\,VI}^{\rm fit}=" + f"{np.log10(N_tot_fit):.2f}",
                        font_size=24, color=COL_FIT).move_to(np.array([2.95, sum_y, 0.0]))

        self.add_hud(sum_t, sum_f)
        self.play(FadeIn(sum_t), FadeIn(sum_f))
        self.wait(2.6)
