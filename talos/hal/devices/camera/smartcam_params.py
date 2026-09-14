"""Parameter tables for the SmartCamApi camera (Zeiss Axiocam 202/208).

Pure data — API facts extracted from ZEN's own SmartCam wrapper (interop
knowledge, see docs/SMARTCAM_API.md). The integer parameter IDs are the
ordinal values of ZEN's ``ParameterKey`` enum, confirmed by ZEN's own log
lines ("WhiteBalance ID is 52", "LedWavelength ID is 72").

All values here are fallbacks: the authoritative per-camera values come from
``ApiCam_GetParameterMetadata`` at runtime.
"""

from __future__ import annotations

from enum import IntEnum


class ParamKey(IntEnum):
    """SmartCamApi parameter IDs (ordinal values — do not reorder)."""

    IllegalParameterKey = 0
    AcquisitionDelay = 1
    AcquisitionMode = 2
    AutoExposure = 3
    AutoGain = 4
    AutoWhiteBalance = 5
    Binning = 6
    BinningExposureDependency = 7
    BinningList = 8
    BlackLevel = 9
    Brightness = 10
    CameraBias = 11
    CameraBitDepth = 12
    CameraModel = 13
    CameraOrientation = 14
    CameraPixelMaximum = 15
    CameraPixelMinimum = 16
    CameraPixelDistance = 17
    CameraPixelType = 18
    CameraSerialNumber = 19
    CameraSize = 20
    CameraTimeout = 21
    CaptureMode = 22
    ColorCorrection = 23
    ColorCorrectionMatrix = 24
    ColorConversionMode = 25
    ColorMode = 26
    ColorSaturation = 27
    ColorSensor = 28
    ColorTemperature = 29
    Contrast = 30
    ExposureGain = 31
    ExposureTime = 32
    FirmwareVersion = 33
    Frame = 34
    FrameValidationMode = 35
    Gamma = 36
    HDRMode = 37
    MultiChannelMode = 38
    NoiseCorrection = 39
    NoiseCorrectionParameters = 40
    ReadoutSpeed = 41
    Resolution = 42
    Sharpness = 43
    SharpnessCorrection = 44
    TargetTemperature = 45
    Temperature = 46
    TemperatureState = 47
    TransferFormat = 48
    TransferQuality = 49
    TriggerMode = 50
    ValidPixelMaximum = 51
    WhiteBalance = 52
    LightManager = 53
    TLIllumination = 54
    RLIllumination = 55
    RLOn = 56
    StageXY = 57
    FocusZ = 58
    TriggerAutoFocus = 59
    CameraAdapter = 60
    ReflectorPosition = 61
    ReflectorCount = 62
    ReflectorMatId = 63
    ObjectivePosition = 64
    ObjectiveCount = 65
    ObjectiveMatId = 66
    ObjectiveMagnification = 67
    LedCount = 68
    LedOn = 69
    LedSelected = 70
    LedBrightness = 71
    LedWaveLength = 72


class ApiError(IntEnum):
    NoError = 0
    InvalidError = 1
    CameraIsOpen = 2
    CameraIndexError = 3
    CameraHandleError = 4
    CameraIsClosed = 5
    EventHandlerNotFound = 6
    EventHandlerIsNull = 7
    InvalidImageCount = 8
    ImageBufferSizeError = 9
    LibraryNotInitialized = 10
    InvalidMetadataType = 11
    MetadataValueNotDefined = 12
    InvalidParameterKey = 13
    InvalidEnumIndex = 14
    PointerIsNull = 15
    ParameterIsReadOnly = 16
    InvalidParameterType = 17
    InvalidParameterValue = 18
    InvalidFunction = 19
    ImageNotReady = 20
    AcquisitionTimeout = 21
    UnexpectedError = 22


class MetadataType(IntEnum):
    Type = 0
    EnumValueType = 1
    Access = 2
    Count = 3
    Minimum = 4
    Maximum = 5
    Increment = 6
    Default = 7
    CanModifyWhileStreaming = 8


class ApiEvent(IntEnum):
    IllegalEvent = 0
    BeginExposure = 1
    EndExposure = 2
    FunctionCompleted = 3
    BufferOverflow = 4
    ImageAcquired = 5
    SequenceAcquired = 6
    ParameterChanged = 7
    TemperatureStable = 8
    TemperatureUnstable = 9


# C# enum ordinals (ParameterType)
PARAM_TYPE_NAMES = {
    0: "boolean", 1: "integer", 2: "double", 3: "string", 4: "enum",
    5: "index_boolean", 6: "index_integer", 7: "index_double", 8: "index_string",
}

# Camera pixel type codes (Zeiss.Micro.Imaging.PixelType): the 208c reports 3.
PIXEL_TYPE_NAMES = {0: "unknown", 1: "gray8", 2: "gray16", 3: "bgr24", 4: "bgr48"}

# Transfer format values (TransferFormat key)
TRANSFER_FORMATS = {0: "nv12", 1: "mjpeg"}
# Resolution values (Resolution key)
RESOLUTIONS = {0: (3840, 2160), 1: (1920, 1080)}
# Auto modes (AutoExposure / AutoWhiteBalance / AutoGain)
AUTO_OFF, AUTO_ON, AUTO_ONCE = 0, 1, 2

# Canonical TALOS white-balance strings -> AutoWhiteBalance values
WB_NAMES = {"Off": AUTO_OFF, "Continuous": AUTO_ON, "Once": AUTO_ONCE}

# White-balance presets applied in software by ZEN (R, G, B factors).
WB_PRESETS = {
    3200: (0.94722047792888, 1.0, 0.48690660662247),
    5500: (0.62657004317365, 1.0, 0.55448148303058),
}

# Canonical TALOS property -> (ParamKey, ctype kind, fallback range, default).
# exposure_us is converted to ms (camera unit) by the backend.
PROPERTIES: dict[str, dict] = {
    "exposure_us": {"key": ParamKey.ExposureTime, "kind": "double",
                    "range": (61.0, 1_000_000.0), "default": 20_000.0,
                    "scale": 1000.0},  # camera uses ms
    "gain": {"key": ParamKey.ExposureGain, "kind": "double",
             "range": (1.0, 22.0), "default": 4.0},
    "color_mode": {"key": ParamKey.ColorMode, "kind": "int",
                   "range": (0, 1), "default": 1},
    "white_balance": {"key": ParamKey.AutoWhiteBalance, "kind": "int",
                      "range": (0, 2), "default": 1},
    "auto_exposure": {"key": ParamKey.AutoExposure, "kind": "int",
                      "range": (0, 2), "default": 0},
    "color_temperature": {"key": ParamKey.ColorTemperature, "kind": "int",
                          "range": (1500, 10000), "default": 5500},
    "gamma": {"key": ParamKey.Gamma, "kind": "double",
              "range": (0.1, 3.0), "default": 1.0},
    "black_level": {"key": ParamKey.BlackLevel, "kind": "int",
                    "range": (0, 4095), "default": 200},
    "resolution": {"key": ParamKey.Resolution, "kind": "int",
                   "range": (0, 1), "default": 1},
    "transfer_format": {"key": ParamKey.TransferFormat, "kind": "int",
                        "range": (0, 1), "default": 0},
    "transfer_quality": {"key": ParamKey.TransferQuality, "kind": "int",
                         "range": (1, 99), "default": 90},
    "sharpness_correction": {"key": ParamKey.SharpnessCorrection, "kind": "int",
                             "range": (0, 1), "default": 1},
    "noise_correction": {"key": ParamKey.NoiseCorrection, "kind": "int",
                         "range": (0, 1), "default": 1},
}

# ZEN's AfterInitialize values applied when apply_defaults is enabled
# (color + fast exposure — the values that give ZEN its 20 fps color live).
ZEN_DEFAULTS = {
    "color_mode": 1,
    "exposure_us": 20_000.0,
    "gain": 4.0,
    "white_balance": 1,
    "resolution": 1,
    "transfer_format": 0,
    "gamma": 0.45,
    "sharpness_correction": 1,
    "noise_correction": 1,
}


def clamp(name: str, value: float) -> float:
    lo, hi = PROPERTIES[name]["range"]
    return min(max(float(value), lo), hi)
