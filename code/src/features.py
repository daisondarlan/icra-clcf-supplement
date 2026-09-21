"""
Shared geometry and feature-extraction code for the landing classifier.

Kept as one module so that the simulated-scene validation
(simulate_depth_scenes.py) runs the exact same feature-extraction functions
as the deployed classifier, not a re-implementation of them -- otherwise a
simulation "validating" the pipeline would really just be validating a
second, independent copy of it.
"""
import math
import numpy as np

# Camera / ROI geometry -- must match the physical depth-camera setup used
# for data collection.
HEIGHT_M = 2.5           # nominal altitude (m), used only for warm-up sizing
AREA_SIZE_M = 0.762       # physical ROI side length (m), halved from 1.524 m (5 ft) for indoor collection
RES_X, RES_Y = 640, 480
FOV_X, FOV_Y = 86, 57     # degrees, matching the depth camera's published specs
OBSTACLE_THRESH = 0.05    # m, deviation-from-mean threshold at the calibration altitude

MIN_VALID_FRACTION = 0.5
MIN_ALTITUDE_M = 0.3
MAX_ALTITUDE_M = 15.0
ALTITUDE_PATCH_FRAC = 0.1

# Depth-noise calibration (see simulate_depth_scenes.py and the paper's
# Check 4 section): sigma_Z = Z^2 / NOISE_K, fit from Session E
# (session_E.csv) -- the one session not flagged as data-quality
# compromised -- mean altitude 2.6346 m, mean flatness_std 0.0316 m. A
# synthetic sweep showed OBSTACLE_THRESH miscalibrated by this noise growth
# beyond the calibration altitude (obstacle_ratio inflated from noise alone,
# not real obstacles); scaling the threshold up past that altitude, floored
# at OBSTACLE_THRESH below it, fixed the effect without making the threshold
# more sensitive to ordinary terrain unevenness at low altitude. This is a
# single-point calibration and should be redone once more verified sessions
# exist.
CALIBRATION_ALTITUDE_M = 2.6346
NOISE_K = 219.41


def scaled_obstacle_thresh(altitude_m):
    return max(OBSTACLE_THRESH, OBSTACLE_THRESH * (altitude_m / CALIBRATION_ALTITUDE_M) ** 2)


def estimate_altitude(depth_image):
    """Live altitude from a small fixed-size central patch (not ROI-dependent)."""
    sx = max(2, int(RES_X * ALTITUDE_PATCH_FRAC))
    sy = max(2, int(RES_Y * ALTITUDE_PATCH_FRAC))
    cx, cy = RES_X // 2, RES_Y // 2
    patch = depth_image[cy - sy // 2: cy + sy // 2, cx - sx // 2: cx + sx // 2]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return None
    altitude = float(np.mean(valid))
    if not (MIN_ALTITUDE_M <= altitude <= MAX_ALTITUDE_M):
        return None
    return altitude


def pixel_pitch_m(altitude_m):
    """Real-world size (metres) of one pixel at the given altitude, along x
    and y. Shared by ROI sizing and fit_plane_angle so both treat the
    pixel grid the same way -- fit_plane_angle needs this to convert a
    fitted slope in "metres of Z per pixel" into a real physical slope in
    "metres of Z per metre", which is what an angle actually means."""
    px = (2 * altitude_m * math.tan(math.radians(FOV_X / 2))) / RES_X
    py = (2 * altitude_m * math.tan(math.radians(FOV_Y / 2))) / RES_Y
    return px, py


def get_crop_indices_for_altitude(altitude_m):
    """Size the ROI in pixels from the *live measured* altitude, not a fixed
    nominal constant -- keeps the ROI's real-world footprint (and therefore
    flatness/obstacle-ratio/angle) comparable throughout descent."""
    px, py = pixel_pitch_m(altitude_m)
    w = min(RES_X, max(1, int(AREA_SIZE_M / px)))
    h = min(RES_Y, max(1, int(AREA_SIZE_M / py)))
    x0 = RES_X // 2 - w // 2
    y0 = RES_Y // 2 - h // 2
    return x0, y0, w, h


def get_crop_indices():
    """Fixed-altitude ROI, kept only for the warm-up phase before the first
    real altitude reading is available (and as the ablation baseline in the
    simulated-scene sweep)."""
    return get_crop_indices_for_altitude(HEIGHT_M)


def fit_plane_angle(z, min_valid_points, px=None, py=None):
    """px, py: real-world metres-per-pixel at the patch's altitude (from
    pixel_pitch_m). Required for a physically correct angle -- without them,
    the plane is fit in raw pixel-index units, so the fitted slope (metres
    of Z per PIXEL) is orders of magnitude smaller than the real slope
    regardless of true tilt, and the reported angle is a massive
    underestimate (e.g. a real ~10 deg tilt reports as ~0.03 deg). px/py
    default to None only to keep this callable in tests that don't care
    about absolute angle."""
    ys, xs = np.mgrid[:z.shape[0], :z.shape[1]]
    valid = np.isfinite(z) & (z > 0)
    if valid.sum() < max(3, min_valid_points):
        return None
    X = np.vstack([xs[valid], ys[valid], np.ones(valid.sum())]).T
    Y = z[valid]
    a, b, _ = np.linalg.lstsq(X, Y, rcond=None)[0]
    if px:
        a = a / px
    if py:
        b = b / py
    normal = np.array([-a, -b, 1.0])
    normal /= np.linalg.norm(normal)
    return math.degrees(math.acos(np.clip(normal[2], -1, 1)))


def compute_flatness_std(z, min_valid_points):
    vals = z[np.isfinite(z) & (z > 0)]
    if vals.size < min_valid_points:
        return None
    return float(np.std(vals))


def compute_obstacle_ratio(z, thresh, min_valid_points):
    vals = z[np.isfinite(z) & (z > 0)]
    if vals.size < min_valid_points:
        return None
    m = vals.mean()
    return float(((np.abs(vals - m) > thresh).sum()) / vals.size)


def compute_metrics_from_depth_image(depth_image):
    """Altitude-adaptive feature extraction from an already-scaled (metres)
    depth image: measure altitude first, size the ROI for that altitude, then
    compute geometric features inside it. Returns
    (angle, flatness, obstacle, altitude, crop_params); any of the first
    three -- or all of them -- are None if the frame can't be trusted.
    """
    altitude = estimate_altitude(depth_image)
    if altitude is None:
        return None, None, None, None, get_crop_indices()

    crop_params = get_crop_indices_for_altitude(altitude)
    x0, y0, w, h = crop_params
    crop = depth_image[y0:y0 + h, x0:x0 + w]
    min_valid_points = int(MIN_VALID_FRACTION * w * h)
    px, py = pixel_pitch_m(altitude)

    angle = fit_plane_angle(crop, min_valid_points, px, py)
    flatness = compute_flatness_std(crop, min_valid_points)
    obstacle = compute_obstacle_ratio(crop, scaled_obstacle_thresh(altitude), min_valid_points)
    return angle, flatness, obstacle, altitude, crop_params
