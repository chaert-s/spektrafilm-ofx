#!/usr/bin/env python3
"""Generate native C++ profile tables from Python profile JSON files."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path


OFX_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("SPEKTRAFILM_DATA_DIR", OFX_ROOT / "Resources" / "data"))
os.environ.setdefault("SPEKTRAFILM_DATA_DIR", str(DATA_DIR))
PROFILE_DIR = Path(os.environ.get("SPEKTRAFILM_PROFILE_DIR", DATA_DIR / "profiles"))
ARCHIVE_PROFILE_DIR = Path(os.environ.get("SPEKTRAFILM_ARCHIVE_PROFILE_DIR", DATA_DIR / "profiles" / "archive"))
ST2065_2_DIR = DATA_DIR / "standards" / "smpte_st_2065_2"
ST2065_2_APD_RESPONSIVITIES_CSV = "st2065-2a-2020.csv"
ST2065_2_INFLUX_SPECTRUM_CSV = "st2065-2b-2020.csv"
HANATOS_LUT_PATH = DATA_DIR / "luts" / "spectral_upsampling" / "irradiance_xy_tc.npy"
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/spektrafilm-mpl-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/spektrafilm-xdg-cache")

import numpy as np
import colour
import scipy.interpolate
import scipy.special
from ofx_stock_lists import DEFAULT_FILM_INDEX, DEFAULT_PAPER_INDEX, FILMS, PAPERS

SPECTRAL_SHAPE = colour.SpectralShape(380, 780, 5)
STANDARD_OBSERVER_CMFS = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(SPECTRAL_SHAPE)
MALLETT2019_BASIS = colour.recovery.MSDS_BASIS_FUNCTIONS_sRGB_MALLETT2019.copy().align(SPECTRAL_SHAPE)

COLOR_DECODE_MIN = -0.125
COLOR_DECODE_MAX = 1.5
COLOR_ENCODE_MIN = -0.25
COLOR_ENCODE_MAX = 64.0
COLOR_LUT_SIZE = 4096

TRANSFER_LINEAR = 0
TRANSFER_LUT = 1
TRANSFER_SRGB = 2
TRANSFER_GAMMA = 3
TRANSFER_PROPHOTO = 4

_OKLAB_M1 = np.asarray(
    [
        [0.8189330101, 0.3618667424, -0.1288597137],
        [0.0329845436, 0.9293118715, 0.0361456387],
        [0.0482003018, 0.2643662691, 0.6338517070],
    ],
    dtype=np.float64,
)
_OKLAB_M2 = np.asarray(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float64,
)
_OKLAB_M1_INV = np.linalg.inv(_OKLAB_M1)
_OKLAB_M2_INV = np.linalg.inv(_OKLAB_M2)


def _unique_numeric_csv(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    unique_index = np.unique(data[:, 0], return_index=True)[1]
    return data[unique_index, :]


def _load_dichroic_filters(wavelengths: np.ndarray, brand: str = "thorlabs") -> np.ndarray:
    filters = np.zeros((np.size(wavelengths), 3))
    for index, channel in enumerate(("c", "m", "y")):
        path = DATA_DIR / "filters" / "dichroics" / brand / f"filter_{channel}.csv"
        data = _unique_numeric_csv(path)
        filters[:, index] = scipy.interpolate.Akima1DInterpolator(data[:, 0], data[:, 1] / 100.0)(wavelengths)
    return filters


def _load_filter(
    wavelengths: np.ndarray,
    name: str = "KG3",
    brand: str = "schott",
    filter_type: str = "heat_absorbing",
    percent_transmittance: bool = False,
) -> np.ndarray:
    path = DATA_DIR / "filters" / filter_type / brand / f"{name}.csv"
    data = _unique_numeric_csv(path)
    scale = 100.0 if percent_transmittance else 1.0
    return scipy.interpolate.Akima1DInterpolator(data[:, 0], data[:, 1] / scale)(wavelengths)


def read_neutral_print_filters() -> dict:
    with (DATA_DIR / "filters" / "neutral_print_filters.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _create_combined_dichroic_filter(
    wavelengths: np.ndarray,
    transitions: tuple[float, float, float, float] = (12.0, 8.0, 8.0, 8.0),
    edges: tuple[float, float, float, float] = (516.0, 500.0, 610.0, 607.0),
) -> np.ndarray:
    filters = np.zeros((np.size(wavelengths), 3))
    filters[:, 2] = scipy.special.erf((wavelengths - edges[0]) / transitions[0])
    filters[:, 1][wavelengths <= 550] = -scipy.special.erf((wavelengths[wavelengths <= 550] - edges[1]) / transitions[1])
    filters[:, 1][wavelengths > 550] = scipy.special.erf((wavelengths[wavelengths > 550] - edges[2]) / transitions[2])
    filters[:, 0] = -scipy.special.erf((wavelengths - edges[3]) / transitions[3])
    return filters * 0.5 + 0.5


class _FilterSet:
    def __init__(self, filters: np.ndarray):
        self.filters = filters


_SPECTRAL_WAVELENGTHS = np.asarray(SPECTRAL_SHAPE.wavelengths, dtype=float)
custom_dichroic_filters = _FilterSet(_create_combined_dichroic_filter(_SPECTRAL_WAVELENGTHS))
_SCHOTT_KG3_HEAT_FILTER = _load_filter(_SPECTRAL_WAVELENGTHS, name="KG3", filter_type="heat_absorbing", brand="schott")
_GENERIC_LENS_TRANSMISSION = _load_filter(
    _SPECTRAL_WAVELENGTHS,
    name="canon_24_f28_is",
    filter_type="lens_transmission",
    brand="canon",
    percent_transmittance=True,
)


def _apply_filter(illuminant: np.ndarray, transmittance: np.ndarray, value: float = 1.0) -> np.ndarray:
    return illuminant * (1.0 - (1.0 - transmittance) * value)


def _black_body_spectrum(temperature: float) -> colour.SpectralDistribution:
    values = colour.colorimetry.blackbody_spectral_radiance(SPECTRAL_SHAPE.wavelengths * 1e-9, temperature)
    return colour.SpectralDistribution(values, domain=SPECTRAL_SHAPE)


def standard_illuminant(illuminant_type: str = "D65", return_class: bool = False):
    if illuminant_type.startswith("BB"):
        spectral_intensity = _black_body_spectrum(float(illuminant_type[2:]))
    elif illuminant_type == "T":
        spectral_intensity = colour.SDS_LIGHT_SOURCES["Incandescent"].copy().align(SPECTRAL_SHAPE)
    elif illuminant_type == "K75P":
        spectral_intensity = colour.SDS_LIGHT_SOURCES["Kinoton 75P"].copy().align(SPECTRAL_SHAPE)
    elif illuminant_type == "TH-KG3":
        spectral_intensity = _black_body_spectrum(3400)
        spectral_intensity.values = _apply_filter(spectral_intensity.values, _SCHOTT_KG3_HEAT_FILTER)
    elif illuminant_type == "TH-KG3-L":
        spectral_intensity = _black_body_spectrum(3400)
        spectral_intensity.values = _apply_filter(spectral_intensity.values, _SCHOTT_KG3_HEAT_FILTER)
        spectral_intensity.values = _apply_filter(spectral_intensity.values, _GENERIC_LENS_TRANSMISSION)
    else:
        spectral_intensity = colour.SDS_ILLUMINANTS[illuminant_type].copy().align(SPECTRAL_SHAPE)

    spectral_intensity.name = illuminant_type
    normalization = np.sum(spectral_intensity.values) / np.size(SPECTRAL_SHAPE.wavelengths)
    spectral_intensity.values = spectral_intensity.values / normalization
    return spectral_intensity if return_class else spectral_intensity[:]


def _illuminant_to_xy(illuminant_label: str) -> np.ndarray:
    illuminant = standard_illuminant(illuminant_label)
    xyz = np.zeros(3)
    cmfs = STANDARD_OBSERVER_CMFS[:]
    for index in range(3):
        xyz[index] = np.sum(illuminant * cmfs[:, index])
    return xyz[0:2] / np.sum(xyz)


HALATION_PRESETS = {
    ("still", "strong"): {"sigma_h": (65.0, 65.0, 65.0), "strength": (0.015, 0.005, 0.0)},
    ("still", "weak"): {"sigma_h": (65.0, 65.0, 65.0), "strength": (0.08, 0.02, 0.0)},
    ("still", "no"): {"sigma_h": (65.0, 65.0, 65.0), "strength": (0.30, 0.10, 0.015)},
    ("cine", "strong"): {"sigma_h": (50.0, 50.0, 50.0), "strength": (0.015, 0.005, 0.0)},
    ("cine", "weak"): {"sigma_h": (50.0, 50.0, 50.0), "strength": (0.08, 0.02, 0.0)},
    ("cine", "no"): {"sigma_h": (50.0, 50.0, 50.0), "strength": (0.30, 0.10, 0.015)},
}

COLOR_SPACES = [
    {
        "key": "arri_logc4",
        "label": "ARRI LogC4",
        "matrix_space": "ARRI Wide Gamut 4",
        "ofx": "arri_logc4",
        "transfer": TRANSFER_LUT,
        "decode": lambda x: colour.models.log_decoding_ARRILogC4(x),
        "encode": lambda x: colour.models.log_encoding_ARRILogC4(x),
    },
    {
        "key": "arri_logc3_ei800",
        "label": "ARRI LogC3 EI800",
        "matrix_space": "ARRI Wide Gamut 3",
        "ofx": "arri_logc3_ei800",
        "transfer": TRANSFER_LUT,
        "decode": lambda x: colour.models.log_decoding_ARRILogC3(x, EI=800),
        "encode": lambda x: colour.models.log_encoding_ARRILogC3(x, EI=800),
    },
    {
        "key": "bmdfilm_widegamut_gen5",
        "label": "BMDFilm WideGamut Gen5",
        "matrix_space": "Blackmagic Wide Gamut",
        "ofx": "bmdfilm_widegamut_gen5",
        "transfer": TRANSFER_LUT,
        "decode": lambda x: colour.models.oetf_inverse_BlackmagicFilmGeneration5(x),
        "encode": lambda x: colour.models.oetf_BlackmagicFilmGeneration5(x),
    },
    {
        "key": "davinci_intermediate_widegamut",
        "label": "DaVinci Intermediate WideGamut",
        "matrix_space": "DaVinci Wide Gamut",
        "ofx": "davinci_intermediate_widegamut",
        "transfer": TRANSFER_LUT,
        "decode": lambda x: colour.models.oetf_inverse_DaVinciIntermediate(x),
        "encode": lambda x: colour.models.oetf_DaVinciIntermediate(x),
    },
    {
        "key": "log3g10_redwidegamutrgb",
        "label": "RED Log3G10 REDWideGamutRGB",
        "matrix_space": "REDWideGamutRGB",
        "ofx": "log3g10_redwidegamutrgb",
        "transfer": TRANSFER_LUT,
        "decode": lambda x: colour.models.log_decoding_Log3G10(x),
        "encode": lambda x: colour.models.log_encoding_Log3G10(x),
    },
    {
        "key": "slog3_sgamut3",
        "label": "Sony S-Log3 S-Gamut3",
        "matrix_space": "S-Gamut3",
        "ofx": "slog3_sgamut3",
        "transfer": TRANSFER_LUT,
        "decode": lambda x: colour.models.log_decoding_SLog3(x),
        "encode": lambda x: colour.models.log_encoding_SLog3(x),
    },
    {
        "key": "slog3_sgamut3cine",
        "label": "Sony S-Log3 S-Gamut3.Cine",
        "matrix_space": "S-Gamut3.Cine",
        "ofx": "slog3_sgamut3cine",
        "transfer": TRANSFER_LUT,
        "decode": lambda x: colour.models.log_decoding_SLog3(x),
        "encode": lambda x: colour.models.log_encoding_SLog3(x),
    },
    {
        "key": "canonlog2_cinemagamut_d55",
        "label": "Canon Log2 CinemaGamut D55",
        "matrix_space": "Cinema Gamut",
        "ofx": "canonlog2_cinemagamut_d55",
        "transfer": TRANSFER_LUT,
        "decode": lambda x: colour.models.log_decoding_CanonLog2(x),
        "encode": lambda x: colour.models.log_encoding_CanonLog2(x),
    },
    {
        "key": "canonlog3_cinemagamut_d55",
        "label": "Canon Log3 CinemaGamut D55",
        "matrix_space": "Cinema Gamut",
        "ofx": "canonlog3_cinemagamut_d55",
        "transfer": TRANSFER_LUT,
        "decode": lambda x: colour.models.log_decoding_CanonLog3(x),
        "encode": lambda x: colour.models.log_encoding_CanonLog3(x),
    },
    {
        "key": "vlog_vgamut",
        "label": "Panasonic V-Log V-Gamut",
        "matrix_space": "V-Gamut",
        "ofx": "vlog_vgamut",
        "transfer": TRANSFER_LUT,
        "decode": lambda x: colour.models.log_decoding_VLog(x),
        "encode": lambda x: colour.models.log_encoding_VLog(x),
    },
    {"key": "aces2065_1", "label": "ACES2065-1", "matrix_space": "ACES2065-1", "ofx": "ACES2065-1", "transfer": TRANSFER_LINEAR},
    {"key": "acescg", "label": "ACEScg", "matrix_space": "ACEScg", "ofx": "ACEScg", "transfer": TRANSFER_LINEAR},
    {
        "key": "acescct",
        "label": "ACEScct",
        "matrix_space": "ACEScg",
        "ofx": "ACEScct",
        "transfer": TRANSFER_LUT,
        "decode": lambda x: colour.models.log_decoding_ACEScct(x),
        "encode": lambda x: colour.models.log_encoding_ACEScct(x),
    },
    {
        "key": "acescc",
        "label": "ACEScc",
        "matrix_space": "ACEScg",
        "ofx": "ACEScc",
        "transfer": TRANSFER_LUT,
        "decode": lambda x: colour.models.log_decoding_ACEScc(x),
        "encode": lambda x: colour.models.log_encoding_ACEScc(x),
    },
    {"key": "lin_rec2020", "label": "Linear Rec.2020", "matrix_space": "ITU-R BT.2020", "ofx": "lin_rec2020", "transfer": TRANSFER_LINEAR},
    {"key": "lin_rec709", "label": "Linear Rec.709", "matrix_space": "ITU-R BT.709", "ofx": "lin_rec709_srgb", "transfer": TRANSFER_LINEAR},
    {"key": "lin_p3d65", "label": "Linear P3-D65", "matrix_space": "P3-D65", "ofx": "lin_p3d65", "transfer": TRANSFER_LINEAR},
    {
        "key": "srgb",
        "label": "sRGB",
        "matrix_space": "sRGB",
        "ofx": "srgb_tx",
        "transfer": TRANSFER_SRGB,
        "decode": lambda x: colour.models.cctf_decoding(x, function="sRGB"),
        "encode": lambda x: colour.models.cctf_encoding(x, function="sRGB"),
    },
    {
        "key": "display_p3",
        "label": "Display P3",
        "matrix_space": "Display P3",
        "ofx": "displayp3_display",
        "transfer": TRANSFER_SRGB,
        "decode": lambda x: colour.models.cctf_decoding(x, function="sRGB"),
        "encode": lambda x: colour.models.cctf_encoding(x, function="sRGB"),
    },
    {
        "key": "prophoto_rgb",
        "label": "ProPhoto RGB",
        "matrix_space": "ProPhoto RGB",
        "ofx": "prophoto_rgb",
        "transfer": TRANSFER_PROPHOTO,
        "decode": lambda x: colour.models.cctf_decoding_ProPhotoRGB(x),
        "encode": lambda x: colour.models.cctf_encoding_ProPhotoRGB(x),
    },
    {
        "key": "adobe_rgb_1998",
        "label": "Adobe RGB (1998)",
        "matrix_space": "Adobe RGB (1998)",
        "ofx": "adobe_rgb_1998",
        "transfer": TRANSFER_GAMMA,
        "transfer_param": 2.19921875,
        "decode": lambda x: np.sign(x) * (np.abs(x) ** 2.19921875),
        "encode": lambda x: np.sign(x) * (np.abs(x) ** (1.0 / 2.19921875)),
    },
    {
        "key": "dci_p3",
        "label": "DCI-P3",
        "matrix_space": "DCI-P3",
        "ofx": "p3_dci_display",
        "transfer": TRANSFER_GAMMA,
        "transfer_param": 2.6,
        "decode": lambda x: np.sign(x) * (np.abs(x) ** 2.6),
        "encode": lambda x: np.sign(x) * (np.abs(x) ** (1.0 / 2.6)),
    },
    {
        "key": "p3d65_gamma22",
        "label": "P3-D65 Gamma 2.2",
        "matrix_space": "P3-D65",
        "ofx": "g22_p3d65_tx",
        "transfer": TRANSFER_GAMMA,
        "transfer_param": 2.2,
        "decode": lambda x: np.sign(x) * (np.abs(x) ** 2.2),
        "encode": lambda x: np.sign(x) * (np.abs(x) ** (1.0 / 2.2)),
    },
    {
        "key": "p3d65_gamma26",
        "label": "P3-D65 Gamma 2.6",
        "matrix_space": "P3-D65",
        "ofx": "g26_p3d65_tx",
        "transfer": TRANSFER_GAMMA,
        "transfer_param": 2.6,
        "decode": lambda x: np.sign(x) * (np.abs(x) ** 2.6),
        "encode": lambda x: np.sign(x) * (np.abs(x) ** (1.0 / 2.6)),
    },
    {
        "key": "rec709_gamma22",
        "label": "Rec.709 Gamma 2.2",
        "matrix_space": "ITU-R BT.709",
        "ofx": "g22_rec709_tx",
        "transfer": TRANSFER_GAMMA,
        "transfer_param": 2.2,
        "decode": lambda x: np.sign(x) * (np.abs(x) ** 2.2),
        "encode": lambda x: np.sign(x) * (np.abs(x) ** (1.0 / 2.2)),
    },
    {
        "key": "rec709_gamma24",
        "label": "Rec.709 Gamma 2.4",
        "matrix_space": "ITU-R BT.709",
        "ofx": "g24_rec709_tx",
        "transfer": TRANSFER_GAMMA,
        "transfer_param": 2.4,
        "decode": lambda x: np.sign(x) * (np.abs(x) ** 2.4),
        "encode": lambda x: np.sign(x) * (np.abs(x) ** (1.0 / 2.4)),
    },
    {
        "key": "flog_fgamut",
        "label": "Fujifilm F-Log F-Gamut",
        "matrix_space": "F-Gamut",
        "ofx": "flog_fgamut",
        "transfer": TRANSFER_LUT,
        "decode": lambda x: colour.models.log_decoding_FLog(x),
        "encode": lambda x: colour.models.log_encoding_FLog(x),
    },
    {
        "key": "flog2_fgamut",
        "label": "Fujifilm F-Log2 F-Gamut",
        "matrix_space": "F-Gamut",
        "ofx": "flog2_fgamut",
        "transfer": TRANSFER_LUT,
        "decode": lambda x: colour.models.log_decoding_FLog2(x),
        "encode": lambda x: colour.models.log_encoding_FLog2(x),
    },
    {
        "key": "flog2c_fgamut_c",
        "label": "Fujifilm F-Log2C F-Gamut C",
        "matrix_space": "F-Gamut C",
        "ofx": "flog2c_fgamut_c",
        "transfer": TRANSFER_LUT,
        "decode": lambda x: colour.models.log_decoding_FLog2(x),
        "encode": lambda x: colour.models.log_encoding_FLog2(x),
    },
]


def _load_profile(stock: str) -> dict:
    return json.loads((PROFILE_DIR / f"{stock}.json").read_text(encoding="utf-8"))


def _load_archive_profile(stock: str) -> dict | None:
    path = ARCHIVE_PROFILE_DIR / f"{stock}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _legacy_info_value(profile: dict, stock: str, key: str):
    info = profile.get("info", {})
    if key in info:
        return info[key]
    archived = _load_archive_profile(stock)
    if archived is not None:
        return archived.get("info", {}).get(key)
    return None


def _archived_bandpass_hanatos2025(stock: str) -> list[list[float | None]] | None:
    archived = _load_archive_profile(stock)
    if archived is None:
        return None
    bandpass = archived.get("data", {}).get("bandpass_hanatos2025", [])
    if not bandpass:
        return None
    return [[None if value is None else float(value) for value in row] for row in bandpass]


def _hanatos2026_window_params(profile: dict) -> list[float]:
    values = profile.get("data", {}).get("hanatos2025_adaptation_window_params", [])
    if not values:
        return [0.0, 0.0, 0.0, 0.0]
    if len(values) != 4:
        raise ValueError("hanatos2025_adaptation_window_params must contain four values")
    return [float(value) for value in values]


def _float_literal(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NAN"
    numeric = float(value)
    if abs(numeric) < 1.0e-38:
        return "0.0f"
    literal = f"{numeric:.9g}"
    if "." not in literal and "e" not in literal and "E" not in literal:
        literal += ".0"
    return f"{literal}f"


def _normalized_density_curves(profile: dict) -> list[list[float]]:
    curves = [[float(value) for value in row] for row in profile["data"]["density_curves"]]
    mins = [min(row[channel] for row in curves) for channel in range(3)]
    return [[row[channel] - mins[channel] for channel in range(3)] for row in curves]


def _density_curve_minimum(profile: dict) -> list[float]:
    curves = [[float(value) for value in row] for row in profile["data"]["density_curves"]]
    return [min(row[channel] for row in curves) for channel in range(3)]


def _density_curves(profile: dict) -> list[list[float]]:
    return [[float(value) for value in row] for row in profile["data"]["density_curves"]]


def _matrix_to_rows(matrix: np.ndarray) -> list[list[float]]:
    return [[float(matrix[row, column]) for column in range(matrix.shape[1])] for row in range(matrix.shape[0])]


def _color_space_to_xyz_matrix(matrix_space: str, reference_illuminant: str) -> list[list[float]]:
    reference_xy = _illuminant_to_xy(reference_illuminant)
    columns = []
    for rgb in np.eye(3):
        columns.append(
            colour.RGB_to_XYZ(
                rgb,
                colourspace=matrix_space,
                apply_cctf_decoding=False,
                illuminant=reference_xy,
                chromatic_adaptation_transform="CAT02",
            )
        )
    return _matrix_to_rows(np.stack(columns, axis=1))


def _color_space_to_meter_xyz_matrix(matrix_space: str) -> list[list[float]]:
    columns = []
    for rgb in np.eye(3):
        columns.append(
            colour.RGB_to_XYZ(
                rgb,
                colourspace=matrix_space,
                apply_cctf_decoding=False,
            )
        )
    return _matrix_to_rows(np.stack(columns, axis=1))


def _xy_to_xyz_unit_y(xy: np.ndarray) -> np.ndarray:
    x = xy[..., 0]
    y = xy[..., 1]
    safe_y = np.maximum(y, 1.0e-12)
    return np.stack([x / safe_y, np.ones_like(x), (1.0 - x - y) / safe_y], axis=-1)


def _chromatic_adaptation_matrix(source_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    source_xyz = _xy_to_xyz_unit_y(np.asarray(source_xy, dtype=np.float64))
    target_xyz = _xy_to_xyz_unit_y(np.asarray(target_xy, dtype=np.float64))
    return np.asarray(
        colour.adaptation.matrix_chromatic_adaptation_VonKries(source_xyz, target_xyz, transform="CAT16"),
        dtype=np.float64,
    )


def _oklab_rgb_matrices(matrix_space: str) -> tuple[np.ndarray, np.ndarray]:
    """Output-RGB <-> OkLab LMS matrices used by the Metal gamut compressor.

    Andrea Volpato's current spektrafilm work uses CAT16 for I/O gamut
    compression. The native renderer follows that by adapting each output
    space white to D65 before the OkLab step, then adapting back after the
    inverse transform.
    """
    rgb_to_xyz = np.asarray(_color_space_to_meter_xyz_matrix(matrix_space), dtype=np.float64)
    xyz_to_rgb = np.linalg.inv(rgb_to_xyz)
    source_white = np.asarray(colour.RGB_COLOURSPACES[matrix_space].whitepoint, dtype=np.float64)
    d65_white = np.asarray(colour.RGB_COLOURSPACES["sRGB"].whitepoint, dtype=np.float64)
    cat_to_d65 = _chromatic_adaptation_matrix(source_white, d65_white)
    cat_from_d65 = np.linalg.inv(cat_to_d65)
    rgb_to_oklab_lms = _OKLAB_M1 @ cat_to_d65 @ rgb_to_xyz
    oklab_lms_to_rgb = xyz_to_rgb @ cat_from_d65 @ _OKLAB_M1_INV
    return rgb_to_oklab_lms, oklab_lms_to_rgb


def _reinhard_knee_array(values: np.ndarray, threshold: float, limit: float, power: float) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    mask = out > threshold
    if np.any(mask):
        scale = max(limit - threshold, 1.0e-12)
        x = (out[mask] - threshold) / scale
        y = x / np.power(1.0 + np.power(x, power), 1.0 / power)
        out[mask] = threshold + scale * y
    return out


def _ray_polygon_distance_array(origin: np.ndarray, direction: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    flat = direction.reshape(-1, 2)
    t_min = np.full(flat.shape[0], np.inf, dtype=np.float64)
    edges = polygon[1:] - polygon[:-1]
    for edge_index, edge in enumerate(edges):
        ex, ey = edge
        ax, ay = polygon[edge_index]
        dx = flat[:, 0]
        dy = flat[:, 1]
        denom = dx * ey - dy * ex
        valid = np.abs(denom) > 1.0e-12
        ox = origin[0] - ax
        oy = origin[1] - ay
        safe_denom = np.where(valid, denom, 1.0)
        t = np.where(valid, (-ox * ey + oy * ex) / safe_denom, np.inf)
        s = np.where(valid, (-ox * dy + oy * dx) / safe_denom, np.inf)
        good = valid & (t > 1.0e-9) & (s >= 0.0) & (s <= 1.0)
        t_min = np.where(good & (t < t_min), t, t_min)
    return t_min.reshape(direction.shape[:-1])


def _compress_xy_radial_for_tests(
    xy: np.ndarray,
    white_xy: np.ndarray,
    locus: np.ndarray,
    threshold: float = 0.0,
    limit: float = 1.0,
    power: float = 6.0,
) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    white_xy = np.asarray(white_xy, dtype=np.float64)
    delta = xy - white_xy
    distance = np.linalg.norm(delta, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        safe_distance = np.maximum(distance, 1.0e-12)
        direction = delta / safe_distance[..., None]
        boundary = _ray_polygon_distance_array(white_xy, direction, np.asarray(locus, dtype=np.float64))
        normalized = distance / np.maximum(boundary, 1.0e-12)
        compressed = _reinhard_knee_array(normalized, threshold, limit, power)
        out = white_xy + direction * (compressed * boundary)[..., None]
    return np.where((distance < 1.0e-9)[..., None], xy, out)


def _output_gamut_compression_data() -> list[float]:
    data: list[float] = []
    for space in COLOR_SPACES:
        rgb_to_oklab_lms, oklab_lms_to_rgb = _oklab_rgb_matrices(space["matrix_space"])
        data.extend(float(value) for value in rgb_to_oklab_lms.reshape(-1))
        data.extend(float(value) for value in oklab_lms_to_rgb.reshape(-1))
    return data


def _compress_rgb_oklch_for_tests(rgb: np.ndarray, color_space_index: int = 0) -> np.ndarray:
    rgb_to_oklab_lms, oklab_lms_to_rgb = _oklab_rgb_matrices(COLOR_SPACES[color_space_index]["matrix_space"])
    rgb = np.asarray(rgb, dtype=np.float64)
    lms = np.einsum("ij,...j->...i", rgb_to_oklab_lms, rgb)
    lms_prime = np.sign(lms) * np.power(np.abs(lms), 1.0 / 3.0)
    lab = np.einsum("ij,...j->...i", _OKLAB_M2, lms_prime)
    flat_rgb = rgb.reshape((-1, 3))
    flat_lab = lab.reshape((-1, 3)).copy()
    flat_out = np.empty_like(flat_rgb)

    def rgb_from_lab(lab_value: np.ndarray) -> np.ndarray:
        lms_prime_value = _OKLAB_M2_INV @ lab_value
        return oklab_lms_to_rgb @ (lms_prime_value * lms_prime_value * lms_prime_value)

    for index, (source_rgb, lab_value) in enumerate(zip(flat_rgb, flat_lab)):
        in_bounds = np.all(np.isfinite(source_rgb)) and np.all(source_rgb >= -1.0e-6) and np.all(source_rgb <= 1.0 + 1.0e-6)
        lab_value[0] = float(_reinhard_knee_array(np.asarray([max(lab_value[0], 0.0)]), 0.7, 1.0, 2.2)[0])
        chroma = float(np.hypot(lab_value[1], lab_value[2]))
        if not np.isfinite(chroma) or chroma <= 1.0e-10:
            flat_out[index] = source_rgb if in_bounds else np.clip(
                rgb_from_lab(np.asarray([lab_value[0], 0.0, 0.0], dtype=np.float64)),
                0.0,
                1.0,
            )
            continue
        hue = lab_value[1:] / chroma
        lo = 0.0
        hi = max(chroma, 1.0e-6)
        for _ in range(12):
            candidate = rgb_from_lab(np.asarray([lab_value[0], hue[0] * hi, hue[1] * hi], dtype=np.float64))
            if not (np.all(np.isfinite(candidate)) and np.all(candidate >= -1.0e-6) and np.all(candidate <= 1.0 + 1.0e-6)):
                break
            lo = hi
            hi = min(hi * 2.0, 4.0)
            if hi >= 4.0:
                break
        for _ in range(16):
            mid = (lo + hi) * 0.5
            candidate = rgb_from_lab(np.asarray([lab_value[0], hue[0] * mid, hue[1] * mid], dtype=np.float64))
            if np.all(np.isfinite(candidate)) and np.all(candidate >= -1.0e-6) and np.all(candidate <= 1.0 + 1.0e-6):
                lo = mid
            else:
                hi = mid
        cmax = max(lo, 1.0e-9)
        normalized = chroma / cmax
        compressed_normalized = float(_reinhard_knee_array(np.asarray([normalized]), 0.0, 1.0, 6.0)[0])
        compressed_chroma = min(compressed_normalized * cmax, cmax)
        flat_out[index] = np.clip(
            rgb_from_lab(np.asarray([lab_value[0], hue[0] * compressed_chroma, hue[1] * compressed_chroma], dtype=np.float64)),
            0.0,
            1.0,
        )
    return flat_out.reshape(rgb.shape)


def _color_space_to_srgb_matrix(matrix_space: str) -> list[list[float]]:
    columns = []
    for rgb in np.eye(3):
        columns.append(
            colour.RGB_to_RGB(
                rgb,
                input_colourspace=matrix_space,
                output_colourspace="sRGB",
                apply_cctf_decoding=False,
                apply_cctf_encoding=False,
            )
        )
    return _matrix_to_rows(np.stack(columns, axis=1))


def _input_to_reference_xyz_matrices(reference_illuminant: str) -> list[list[float]]:
    matrices: list[list[float]] = []
    for space in COLOR_SPACES:
        matrices.extend(_color_space_to_xyz_matrix(space["matrix_space"], reference_illuminant))
    return matrices


def _input_to_meter_xyz_matrices() -> list[list[float]]:
    matrices: list[list[float]] = []
    for space in COLOR_SPACES:
        matrices.extend(_color_space_to_meter_xyz_matrix(space["matrix_space"]))
    return matrices


def _input_to_srgb_matrices() -> list[list[float]]:
    matrices: list[list[float]] = []
    for space in COLOR_SPACES:
        matrices.extend(_color_space_to_srgb_matrix(space["matrix_space"]))
    return matrices


def _scan_to_output_rgb_matrices(viewing_illuminant: str) -> list[list[float]]:
    scan_illuminant = np.asarray(standard_illuminant(viewing_illuminant), dtype=float)
    cmfs = np.asarray(STANDARD_OBSERVER_CMFS[:], dtype=float)
    normalization = np.sum(scan_illuminant * cmfs[:, 1], axis=0)
    illuminant_xyz = np.dot(scan_illuminant, cmfs) / normalization
    illuminant_xy = colour.XYZ_to_xy(illuminant_xyz)
    matrices: list[list[float]] = []
    for space in COLOR_SPACES:
        columns = []
        for xyz in np.eye(3):
            columns.append(
                colour.XYZ_to_RGB(
                    xyz,
                    colourspace=space["matrix_space"],
                    illuminant=illuminant_xy,
                    chromatic_adaptation_transform="CAT02",
                    apply_cctf_encoding=False,
                )
            )
        matrices.extend(_matrix_to_rows(np.stack(columns, axis=1)))
    return matrices


def _lut_values(space: dict, key: str, minimum: float, maximum: float) -> list[float]:
    x = np.linspace(minimum, maximum, COLOR_LUT_SIZE, dtype=np.float64)
    if space["transfer"] == TRANSFER_LINEAR:
        y = x
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            y = space[key](x)
    return [float(value) for value in np.nan_to_num(y, nan=0.0, posinf=1.0e10, neginf=-1.0e10)]


def _emit_color_transforms() -> str:
    decode_luts: list[float] = []
    encode_luts: list[float] = []
    transfer_kinds: list[int] = []
    transfer_params: list[float] = []
    label_records: list[str] = []
    for space in COLOR_SPACES:
        decode_luts.extend(_lut_values(space, "decode", COLOR_DECODE_MIN, COLOR_DECODE_MAX))
        encode_luts.extend(_lut_values(space, "encode", COLOR_ENCODE_MIN, COLOR_ENCODE_MAX))
        transfer_kinds.append(int(space["transfer"]))
        transfer_params.append(float(space.get("transfer_param", 0.0)))
        label_records.append(f"  {json.dumps(space['label'])},")

    return "\n\n".join(
        [
            _array("color_decode_luts", decode_luts),
            _array("color_encode_luts", encode_luts),
            _array("input_meter_xyz", _input_to_meter_xyz_matrices()),
            "alignas(16) constexpr uint32_t color_transfer_kinds[] = {"
            + ", ".join(f"{value}u" for value in transfer_kinds)
            + "};",
            _array("color_transfer_params", transfer_params),
            "constexpr const char *color_space_labels[] = {",
            "\n".join(label_records),
            "};",
        ]
    )


def _mallett_basis_illuminant(reference_illuminant: str) -> list[list[float]]:
    illuminant = np.asarray(standard_illuminant(reference_illuminant), dtype=float)
    basis = np.asarray(MALLETT2019_BASIS[:], dtype=float)
    return _matrix_to_rows(basis * illuminant[:, None])


def _mallett_midgray_green(profile: dict, reference_illuminant: str) -> float:
    log_sensitivity = np.asarray(profile["data"]["log_sensitivity"], dtype=float)
    sensitivity = np.nan_to_num(10.0 ** log_sensitivity)
    illuminant = np.asarray(standard_illuminant(reference_illuminant), dtype=float)
    return float(np.sum(illuminant * 0.184 * sensitivity[:, 1]))


def _neutral_print_filter_table() -> list[list[float]]:
    filters = read_neutral_print_filters()
    fallback = [0.0, 65.0, 55.0]
    table: list[list[float]] = []
    for paper in PAPERS:
        for film in FILMS:
            values = filters.get(paper, {}).get("TH-KG3", {}).get(film, fallback)
            table.append([float(value) for value in values])
    return table


def _read_numeric_csv_rows(path: Path, column_count: int) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = [part.strip() for part in line.rstrip("\n").split(",")]
            if len(parts) < column_count:
                continue
            try:
                rows.append([float(parts[i]) for i in range(column_count)])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    return np.asarray(rows, dtype=float)


def _st2065_2_paths() -> tuple[Path, Path]:
    return (
        ST2065_2_DIR / ST2065_2_APD_RESPONSIVITIES_CSV,
        ST2065_2_DIR / ST2065_2_INFLUX_SPECTRUM_CSV,
    )


def academy_printer_density_available() -> bool:
    return all(path.is_file() for path in _st2065_2_paths())


def _missing_st2065_2_paths() -> list[Path]:
    return [path for path in _st2065_2_paths() if not path.is_file()]


def _zero_spectral_rows(channel_count: int) -> list[list[float]]:
    return [
        [0.0 for _ in range(channel_count)]
        for _ in SPECTRAL_SHAPE.wavelengths
    ]


def _zero_spectral_vector() -> list[float]:
    return [0.0 for _ in SPECTRAL_SHAPE.wavelengths]


def _zero_printer_density_neutral_offsets() -> list[list[float]]:
    return [
        [0.0, 0.0, 0.0]
        for _paper in PAPERS
        for _film in FILMS
    ]


def print_missing_st2065_2_disclaimer() -> None:
    missing = _missing_st2065_2_paths()
    if not missing:
        return
    lines = [
        "",
        "================================================================================",
        "IMPORTANT SPEKTRAFILM BUILD NOTICE",
        "SMPTE ST 2065-2 CSV files were not found in the local checkout.",
        "",
        "Academy Printer Density / printer-light mode will be DISABLED in this OFX build.",
        "The plugin will expose only Filtered Enlarger print timing, and printer-point",
        "controls will not be available.",
        "",
        "This is expected for public source builds because the SMPTE CSV files are",
        "licensed standards data and are not redistributed in this repository.",
        "",
        "Missing files:",
    ]
    lines.extend(f"  - {path}" for path in missing)
    lines.extend([
        "================================================================================",
        "",
    ])
    print("\n".join(lines), file=sys.stderr)


def _st2065_2_apd_responsivities() -> list[list[float]]:
    if not academy_printer_density_available():
        return _zero_spectral_rows(3)
    data = _read_numeric_csv_rows(ST2065_2_DIR / ST2065_2_APD_RESPONSIVITIES_CSV, 4)
    wavelengths = data[:, 0]
    values = data[:, 1:4]
    target_wavelengths = np.asarray(SPECTRAL_SHAPE.wavelengths, dtype=float)
    resampled = np.column_stack([
        np.interp(target_wavelengths, wavelengths, values[:, channel], left=0.0, right=0.0)
        for channel in range(3)
    ])
    return _matrix_to_rows(resampled)


def _st2065_2_influx_spectrum() -> list[float]:
    if not academy_printer_density_available():
        return _zero_spectral_vector()
    data = _read_numeric_csv_rows(ST2065_2_DIR / ST2065_2_INFLUX_SPECTRUM_CSV, 2)
    wavelengths = data[:, 0]
    values = data[:, 1]
    target_wavelengths = np.asarray(SPECTRAL_SHAPE.wavelengths, dtype=float)
    return [float(value) for value in np.interp(target_wavelengths, wavelengths, values, left=0.0, right=0.0)]


def _apd_raw_for_film_density(profile: dict, density_cmy: np.ndarray, apd_responsivities: np.ndarray) -> np.ndarray:
    channel_density = np.asarray(profile["data"]["channel_density"], dtype=float)
    base_density = np.asarray(profile["data"]["base_density"], dtype=float)
    density_spectral = channel_density @ density_cmy + base_density
    transmittance = np.where(np.isfinite(density_spectral), 10.0 ** (-density_spectral), 0.0)
    normalization = np.maximum(np.sum(apd_responsivities, axis=0), 1.0e-10)
    return np.sum(apd_responsivities * transmittance[:, None], axis=0) / normalization


def _paper_midscale_neutral_log_exposure(profile: dict) -> np.ndarray:
    stock = profile["info"]["stock"]
    target_density_value = _legacy_info_value(profile, stock, "fitted_cmy_midscale_neutral_density")
    if target_density_value is None:
        raise KeyError(f"Missing fitted_cmy_midscale_neutral_density for paper profile '{stock}'")
    target_density = np.asarray(target_density_value, dtype=float)
    density_curves = np.asarray(profile["data"]["density_curves"], dtype=float)
    log_exposure = np.asarray(profile["data"]["log_exposure"], dtype=float)
    target_log_exposure = np.zeros(3, dtype=float)
    for channel in range(3):
        density = density_curves[:, channel]
        valid = np.isfinite(density) & np.isfinite(log_exposure)
        target_log_exposure[channel] = np.interp(target_density[channel], density[valid], log_exposure[valid])
    return target_log_exposure


def _academy_printer_density_neutral_offset_table() -> list[list[float]]:
    if not academy_printer_density_available():
        return _zero_printer_density_neutral_offsets()
    apd_responsivities = np.asarray(_st2065_2_apd_responsivities(), dtype=float)
    log10_two = math.log10(2.0)
    table: list[list[float]] = []
    for paper in PAPERS:
        paper_profile = _load_profile(paper)
        paper_target_log = _paper_midscale_neutral_log_exposure(paper_profile)
        paper_target_relative_log = paper_target_log - float(np.mean(paper_target_log))
        for film in FILMS:
            film_profile = _load_profile(film)
            film_midscale_value = _legacy_info_value(film_profile, film, "fitted_cmy_midscale_neutral_density")
            if film_midscale_value is None:
                raise KeyError(f"Missing fitted_cmy_midscale_neutral_density for film profile '{film}'")
            film_midscale_density = np.asarray(film_midscale_value, dtype=float)
            film_density_minimum = np.asarray(_density_curve_minimum(film_profile), dtype=float)
            film_density_cmy = np.maximum(film_midscale_density - film_density_minimum, 0.0)
            apd_raw = np.maximum(_apd_raw_for_film_density(film_profile, film_density_cmy, apd_responsivities), 1.0e-10)
            apd_raw_relative_log = np.log10(apd_raw) - float(np.mean(np.log10(apd_raw)))
            neutral_points = 12.0 * (paper_target_relative_log - apd_raw_relative_log) / log10_two
            table.append([float(value) for value in neutral_points])
    return table


def _academy_printer_density_data() -> list[list[float]]:
    return _st2065_2_apd_responsivities() + _academy_printer_density_neutral_offset_table()


def _flatten(values) -> list[float | None]:
    flat: list[float | None] = []
    if isinstance(values, list):
        for value in values:
            flat.extend(_flatten(value))
    else:
        flat.append(values)
    return flat


def _array(name: str, values: list[float] | list[list[float]]) -> str:
    flat = _flatten(values)

    lines = [f"alignas(16) constexpr float {name}[] = {{"]
    for index in range(0, len(flat), 6):
        chunk = ", ".join(_float_literal(value) for value in flat[index : index + 6])
        lines.append(f"  {chunk},")
    lines.append("};")
    return "\n".join(lines)


def _numeric_matrix(profile: dict, key: str) -> list[list[float | None]]:
    return [[None if value is None else float(value) for value in row] for row in profile["data"][key]]


def _numeric_vector(profile: dict, key: str) -> list[float | None]:
    return [None if value is None else float(value) for value in profile["data"][key]]


def _numeric_layers(profile: dict, key: str) -> list[list[list[float | None]]]:
    return [
        [[None if value is None else float(value) for value in layer] for layer in row]
        for row in profile["data"][key]
    ]


def _density_curve_layer_maxima(profile: dict) -> list[list[float]]:
    layers = np.asarray(profile["data"]["density_curves_layers"], dtype=float)
    maxima = np.nanmax(layers, axis=0)
    return [[float(value) for value in row] for row in maxima]


def _halation_preset(info: dict) -> dict[str, tuple[float, float, float]]:
    return HALATION_PRESETS.get(
        (info.get("use", "still"), info.get("antihalation", "weak")),
        {"sigma_h": (65.0, 65.0, 65.0), "strength": (0.05, 0.015, 0.0)},
    )


def _dir_coupler_defaults(info: dict) -> dict[str, tuple[float, ...]]:
    """Mirror the Python runtime's film-specific DIR defaults for native OFX data."""
    if info["type"] == "positive":
        defaults = {
            "same_layer_rgb": (0.12, 0.08, 0.06),
            "r_to_gb": (0.12, 0.06),
            "g_to_rb": (0.08, 0.06),
            "b_to_rg": (0.06, 0.06),
        }
    else:
        defaults = {
            "same_layer_rgb": (0.336, 0.319, 0.273),
            "r_to_gb": (0.353, 0.302),
            "g_to_rb": (0.154, 0.353),
            "b_to_rg": (0.168, 0.226),
        }

    stock = info["stock"]
    if stock == "fujifilm_velvia_100":
        return {
            "same_layer_rgb": (0.108, 0.072, 0.054),
            "r_to_gb": (0.108, 0.054),
            "g_to_rb": (0.072, 0.054),
            "b_to_rg": (0.054, 0.054),
        }
    if stock == "fujifilm_provia_100f":
        return {
            "same_layer_rgb": (0.156, 0.104, 0.078),
            "r_to_gb": (0.156, 0.078),
            "g_to_rb": (0.104, 0.078),
            "b_to_rg": (0.078, 0.078),
        }
    return defaults


def _emit_group(group_name: str, stocks: list[str]) -> str:
    chunks: list[str] = []
    records: list[str] = []
    input_to_srgb = _input_to_srgb_matrices()
    chunks.append(_array(f"{group_name}_input_to_srgb", input_to_srgb))
    for index, stock in enumerate(stocks):
        profile = _load_profile(stock)
        info = profile["info"]
        prefix = f"{group_name}_{index}_{stock}"
        reference_illuminant = info["reference_illuminant"]
        viewing_illuminant = info["viewing_illuminant"]
        wavelengths = _numeric_vector(profile, "wavelengths")
        log_sensitivity = _numeric_matrix(profile, "log_sensitivity")
        bandpass_hanatos2025 = _archived_bandpass_hanatos2025(stock)
        hanatos2026_window_params = _hanatos2026_window_params(profile)
        input_to_xyz = _input_to_reference_xyz_matrices(reference_illuminant)
        reference_illuminant_spectrum = [float(value) for value in standard_illuminant(reference_illuminant)]
        scan_illuminant = [float(value) for value in standard_illuminant(viewing_illuminant)]
        scan_to_output_rgb = _scan_to_output_rgb_matrices(viewing_illuminant)
        mallett_basis_illuminant = _mallett_basis_illuminant(reference_illuminant)
        mallett_midgray_green = _mallett_midgray_green(profile, reference_illuminant)
        log_exposure = [float(value) for value in profile["data"]["log_exposure"]]
        density_curves = _normalized_density_curves(profile) if group_name == "film" else _density_curves(profile)
        density_curve_minimum = _density_curve_minimum(profile)
        density_curve_layers = _numeric_layers(profile, "density_curves_layers")
        density_curve_layer_maxima = _density_curve_layer_maxima(profile)
        halation_preset = _halation_preset(info)
        dir_couplers = _dir_coupler_defaults(info)
        channel_density = _numeric_matrix(profile, "channel_density")
        base_density = _numeric_vector(profile, "base_density")
        chunks.append(_array(f"{prefix}_wavelengths", wavelengths))
        chunks.append(_array(f"{prefix}_log_sensitivity", log_sensitivity))
        if bandpass_hanatos2025 is not None:
            chunks.append(_array(f"{prefix}_bandpass_hanatos2025", bandpass_hanatos2025))
        chunks.append(_array(f"{prefix}_hanatos2026_window_params", hanatos2026_window_params))
        chunks.append(_array(f"{prefix}_reference_illuminant_spectrum", reference_illuminant_spectrum))
        chunks.append(_array(f"{prefix}_input_to_reference_xyz", input_to_xyz))
        chunks.append(_array(f"{prefix}_mallett_basis_illuminant", mallett_basis_illuminant))
        chunks.append(_array(f"{prefix}_log_exposure", log_exposure))
        chunks.append(_array(f"{prefix}_density_curves", density_curves))
        chunks.append(_array(f"{prefix}_channel_density", channel_density))
        chunks.append(_array(f"{prefix}_base_density", base_density))
        chunks.append(_array(f"{prefix}_density_curve_minimum", density_curve_minimum))
        chunks.append(_array(f"{prefix}_density_curve_layers", density_curve_layers))
        chunks.append(_array(f"{prefix}_density_curve_layer_maxima", density_curve_layer_maxima))
        chunks.append(_array(f"{prefix}_halation_strength", [float(value) for value in halation_preset["strength"]]))
        chunks.append(_array(f"{prefix}_halation_first_sigma_um", [float(value) for value in halation_preset["sigma_h"]]))
        chunks.append(_array(f"{prefix}_dir_gamma_same_layer_rgb", [float(value) for value in dir_couplers["same_layer_rgb"]]))
        chunks.append(_array(f"{prefix}_dir_gamma_r_to_gb", [float(value) for value in dir_couplers["r_to_gb"]]))
        chunks.append(_array(f"{prefix}_dir_gamma_g_to_rb", [float(value) for value in dir_couplers["g_to_rb"]]))
        chunks.append(_array(f"{prefix}_dir_gamma_b_to_rg", [float(value) for value in dir_couplers["b_to_rg"]]))
        chunks.append(_array(f"{prefix}_scan_illuminant", scan_illuminant))
        chunks.append(_array(f"{prefix}_scan_to_output_rgb", scan_to_output_rgb))
        bandpass_hanatos2025_ptr = (
            f"{prefix}_bandpass_hanatos2025" if bandpass_hanatos2025 is not None else "nullptr"
        )
        records.append(
            "  {"
            f"{json.dumps(stock)}, "
            f"{json.dumps(info['name'])}, "
            f"{json.dumps(info['type'])}, "
            f"{json.dumps(reference_illuminant)}, "
            f"{len(wavelengths)}u, "
            f"{len(log_exposure)}u, "
            f"{prefix}_wavelengths, "
            f"{prefix}_log_sensitivity, "
            f"{bandpass_hanatos2025_ptr}, "
            f"{prefix}_hanatos2026_window_params, "
            f"{prefix}_reference_illuminant_spectrum, "
            f"{prefix}_input_to_reference_xyz, "
            f"{group_name}_input_to_srgb, "
            f"{prefix}_mallett_basis_illuminant, "
            f"{_float_literal(mallett_midgray_green)}, "
            f"{prefix}_log_exposure, "
            f"{prefix}_density_curves, "
            f"{prefix}_channel_density, "
            f"{prefix}_base_density, "
            f"{prefix}_density_curve_minimum, "
            f"{prefix}_density_curve_layers, "
            f"{prefix}_density_curve_layer_maxima, "
            f"{prefix}_halation_strength, "
            f"{prefix}_halation_first_sigma_um, "
            f"{prefix}_dir_gamma_same_layer_rgb, "
            f"{prefix}_dir_gamma_r_to_gb, "
            f"{prefix}_dir_gamma_g_to_rb, "
            f"{prefix}_dir_gamma_b_to_rg, "
            f"{prefix}_scan_illuminant, "
            f"{prefix}_scan_to_output_rgb"
            "},"
        )
    chunks.append(f"constexpr ProfileCurveSet {group_name}_profiles[] = {{")
    chunks.extend(records)
    chunks.append("};")
    return "\n\n".join(chunks)


def generate() -> str:
    return "\n".join(
        [
            "// Generated by OFX/SpektraFilm/tools/generate_profile_curves.py.",
            "// Source of truth: OFX/SpektraFilm/Resources/data/profiles/*.json.",
            "#include \"SpektraProfileCurves.h\"",
            "",
            "#include <cmath>",
            "",
            "namespace spektrafilm {",
            "namespace {",
            "",
            _emit_group("film", FILMS),
            "",
            _emit_group("paper", PAPERS),
            "",
            _array("standard_observer_cmfs", _matrix_to_rows(np.asarray(STANDARD_OBSERVER_CMFS[:], dtype=float))),
            "",
            _array("th_kg3_illuminant", [float(value) for value in standard_illuminant("TH-KG3")]),
            "",
            _array("custom_enlarger_filters", _matrix_to_rows(np.asarray(custom_dichroic_filters.filters, dtype=float))),
            "",
            _array("neutral_print_filters", _neutral_print_filter_table()),
            "",
            _array("academy_printer_density_responsivities", _st2065_2_apd_responsivities()),
            "",
            _array("academy_printer_density_neutral_offsets", _academy_printer_density_neutral_offset_table()),
            "",
            _array("academy_printer_density_data", _academy_printer_density_data()),
            "",
            _array("academy_printer_density_influx_spectrum", _st2065_2_influx_spectrum()),
            "",
            _emit_color_transforms(),
            "",
            "constexpr HanatosSpectraLutInfo hanatos_lut_info = {192u, 192u, 81u, 2985984u};",
            "",
            "} // namespace",
            "",
            "const ProfileCurveSet *filmProfileCurves(int32_t index) {",
            "  if (index < 0 || index >= static_cast<int32_t>(sizeof(film_profiles) / sizeof(film_profiles[0]))) {",
            "    return nullptr;",
            "  }",
            "  return &film_profiles[index];",
            "}",
            "",
            "const ProfileCurveSet *paperProfileCurves(int32_t index) {",
            "  if (index < 0 || index >= static_cast<int32_t>(sizeof(paper_profiles) / sizeof(paper_profiles[0]))) {",
            "    return nullptr;",
            "  }",
            "  return &paper_profiles[index];",
            "}",
            "",
            "const HanatosSpectraLutInfo &hanatosSpectraLutInfo() {",
            "  return hanatos_lut_info;",
            "}",
            "",
            "const float *inputMeterXyzMatrices() {",
            "  return input_meter_xyz;",
            "}",
            "",
            "const uint32_t *colorTransferKinds() {",
            "  return color_transfer_kinds;",
            "}",
            "",
            "const float *colorTransferParams() {",
            "  return color_transfer_params;",
            "}",
            "",
            "const char *colorSpaceLabel(int32_t index) {",
            "  if (index < 0 || index >= static_cast<int32_t>(kSpektraColorSpaceCount)) {",
            "    return nullptr;",
            "  }",
            "  return color_space_labels[index];",
            "}",
            "",
            "const float *colorDecodeLuts() {",
            "  return color_decode_luts;",
            "}",
            "",
            "const float *colorEncodeLuts() {",
            "  return color_encode_luts;",
            "}",
            "",
            "const float *standardObserverCmfs() {",
            "  return standard_observer_cmfs;",
            "}",
            "",
            "const float *thKg3Illuminant() {",
            "  return th_kg3_illuminant;",
            "}",
            "",
            "const float *customEnlargerFilters() {",
            "  return custom_enlarger_filters;",
            "}",
            "",
            "const float *neutralPrintFilters() {",
            "  return neutral_print_filters;",
            "}",
            "",
            "const float *academyPrinterDensityResponsivities() {",
            "  return academy_printer_density_responsivities;",
            "}",
            "",
            "const float *academyPrinterDensityNeutralOffsets() {",
            "  return academy_printer_density_neutral_offsets;",
            "}",
            "",
            "const float *academyPrinterDensityData() {",
            "  return academy_printer_density_data;",
            "}",
            "",
            "const float *academyPrinterDensityInfluxSpectrum() {",
            "  return academy_printer_density_influx_spectrum;",
            "}",
            "",
            "float colorDecodeLutMin() {",
            f"  return {_float_literal(COLOR_DECODE_MIN)};",
            "}",
            "",
            "float colorDecodeLutMax() {",
            f"  return {_float_literal(COLOR_DECODE_MAX)};",
            "}",
            "",
            "float colorEncodeLutMin() {",
            f"  return {_float_literal(COLOR_ENCODE_MIN)};",
            "}",
            "",
            "float colorEncodeLutMax() {",
            f"  return {_float_literal(COLOR_ENCODE_MAX)};",
            "}",
            "",
            "} // namespace spektrafilm",
            "",
        ]
    )


def generate_counts_header() -> str:
    return "\n".join(
        [
            "// Generated by OFX/SpektraFilm/tools/generate_profile_curves.py.",
            "// Source of truth: OFX/SpektraFilm/tools/ofx_stock_lists.py.",
            "#pragma once",
            "",
            "#include <cstdint>",
            "",
            "#define SPEKTRA_GENERATED_PROFILE_COUNTS 1",
            "",
            "namespace spektrafilm {",
            "",
            f"constexpr uint32_t kSpektraFilmCount = {len(FILMS)}u;",
            f"constexpr uint32_t kSpektraPaperCount = {len(PAPERS)}u;",
            f"constexpr int32_t kSpektraDefaultFilmIndex = {DEFAULT_FILM_INDEX};",
            f"constexpr int32_t kSpektraDefaultPaperIndex = {DEFAULT_PAPER_INDEX};",
            f"constexpr bool kSpektraAcademyPrinterDensityEnabled = {'true' if academy_printer_density_available() else 'false'};",
            "",
            "} // namespace spektrafilm",
            "",
        ]
    )


def write_hanatos_lut(output: Path) -> None:
    import numpy as np

    lut = np.load(HANATOS_LUT_PATH).astype("<f4", copy=False)
    if lut.shape != (192, 192, 81):
        raise ValueError(f"Unexpected Hanatos LUT shape {lut.shape}; expected (192, 192, 81).")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(lut.tobytes(order="C"))


def write_output_gamut_compression_data(output: Path) -> None:
    data = np.asarray(_output_gamut_compression_data(), dtype="<f4")
    expected_count = len(COLOR_SPACES) * 18
    if data.size != expected_count:
        raise ValueError(f"Unexpected output gamut compression data size {data.size}; expected {expected_count}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data.tobytes(order="C"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts-output", type=Path)
    parser.add_argument("--hanatos-output", type=Path)
    parser.add_argument("--output-gamut-compression-output", type=Path)
    args = parser.parse_args()
    print_missing_st2065_2_disclaimer()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generate(), encoding="utf-8")
    print(f"Wrote {args.output}")
    if args.counts_output:
        args.counts_output.parent.mkdir(parents=True, exist_ok=True)
        args.counts_output.write_text(generate_counts_header(), encoding="utf-8")
        print(f"Wrote {args.counts_output}")
    if args.hanatos_output:
        write_hanatos_lut(args.hanatos_output)
        print(f"Wrote {args.hanatos_output}")
    if args.output_gamut_compression_output:
        write_output_gamut_compression_data(args.output_gamut_compression_output)
        print(f"Wrote {args.output_gamut_compression_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
