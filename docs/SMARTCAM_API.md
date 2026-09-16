# SmartCamApi — interoperability notes

API facts needed to drive a Zeiss Axiocam 208 through the `SmartCamApi.dll`
that ships with ZEN / Labscope: entry points, calling conventions, parameter
keys, value ranges and the behaviour of the acquisition calls.

This file records observed API FACTS (signatures, constants, semantics) for
interoperability, with no vendor code reproduced. Everything here was verified
against real hardware on this bench; the numbers that matter are also encoded
as constants in `talos/hal/devices/camera/smartcam_params.py`, which is what the
backend actually uses.

**Scope**: the Axiocam 208 / 208 color family, on the DLL versions that shipped
with the ZEN/Labscope releases current at the time of writing. Treat an ordinal
or a range as a hypothesis to verify on your own camera, not as a spec — see
[CAMERA_208.md](CAMERA_208.md) for the behavioural summary and
[DESIGN.md](DESIGN.md) for where this fits in the bench.

## Calling convention & signatures

All exports are **Cdecl**. Primitives are passed by value and mutable values
**by pointer**. THIS IS WHY THE OLD BLIND SWEEP FAILED: our old code passed
`c_double` BY VALUE where the DLL expects a POINTER to the value. Fix: pass
`ctypes.byref(...)`.

- `ApiLib_InitializeLibrary(ApiOptions options)` — options struct by value:
  `{u16 OptionVersion, u16 OptionSize, u32 OptionFlags}` (flags: 1=OutputDebugMessages)
- `ApiLib_GetLibraryInformation(out ApiInformation)` — `{u16 InfoVersion, InfoSize,
  ApiVersion, MaxStringLength, MaxParameterCount, MaxFunctionCount, MaxEventCount,
  MaxError, ImageHeaderSize}` (all u16, 18 bytes)
- `ApiLib_GetCameraCount(out int)` · `ApiLib_GetErrorDescription(ApiError, StringBuilder)`
  · `ApiLib_FinalizeLibrary()`
- `ApiCam_OpenCamera(int idx, out IntPtr handle)` · `ApiCam_CloseCamera(handle)`
- `ApiCam_GetParameterList(handle, int[] paramList)` — 0-terminated (0=IllegalParameterKey)
- `ApiCam_GetParameterMetadata(handle, key, MetadataType, out int | out double | int[] | IntPtr)`
- `ApiCam_GetEnumParameterMetadata(handle, key, int idx, out int enumValue, StringBuilder desc)`
- `ApiCam_GetParameterDescription(handle, key, StringBuilder)`
- `ApiCam_GetParameterValue(handle, key, out byte | out int | out double | StringBuilder | IntPtr)`
  — IntPtr variant for array params (e.g. `int[2]` CameraSize, `int[4]` quads)
- `ApiCam_SetParameterValue(handle, key, ref byte | ref int | ref double | StringBuilder | IntPtr)`
- `ApiCam_GetAcquisitionBufferSize(handle, int imageCount, out u64 bufferByteSize)`
- `ApiCam_AcquireSingleImage(handle, u64 bufferByteSize, byte[] buffer)`
- `ApiCam_StartSequenceAcquisition(handle, int imageCount, u64 bufferByteSize, byte[] buffer)`
- `ApiCam_GetSequenceImage(handle)` — POLL until NoError (ImageNotReady=20); takes no buffer
  (writes into the buffer given at Start)
- `ApiCam_StartContinuousAcquisition(handle, long bufferByteSize, byte[] buffer)`
- `ApiCam_RaiseSoftwareTrigger(handle)` · `ApiCam_AbortAcquisition(handle)`
- `ApiCam_GetEventList(handle, int[])` · `ApiCam_AddEventHandler(handle, delegate(ApiEvent, IntPtr))`
  · `ApiCam_RemoveEventHandler(...)`
- `ApiCam_GetFunctionList(handle, int[])` · `ApiCam_PerformFunction(CamFunction)`
- `ApiCam_GetImageMetadata(int imageByteSize, byte[] image, out ApiImageMetadata)`

## Error codes (ApiError)

0 NoError · 1 InvalidError · 2 CameraIsOpen · 3 CameraIndexError · 4 CameraHandleError ·
5 CameraIsClosed · 6 EventHandlerNotFound · 7 EventHandlerIsNull · 8 InvalidImageCount ·
9 ImageBufferSizeError · 10 LibraryNotInitialized · 11 InvalidMetadataType ·
12 MetadataValueNotDefined · 13 InvalidParameterKey · 14 InvalidEnumIndex · 15 PointerIsNull ·
16 ParameterIsReadOnly · 17 InvalidParameterType · 18 InvalidParameterValue · 19 InvalidFunction ·
20 ImageNotReady · 21 AcquisitionTimeout · 22 UnexpectedError

## MetadataType (GetParameterMetadata)

0 Type · 1 EnumValueType · 2 Access · 3 Count · 4 Minimum · 5 Maximum · 6 Increment ·
7 Default · 8 CanModifyWhileStreaming

## ParameterType

0 Boolean · 1 Integer · 2 Double · 3 String · 4 Enum · 5 IndexAndBoolean ·
6 IndexAndInteger · 7 IndexAndDouble · 8 IndexAndString

## ParameterKey IDs (0-based ordinals), cross-checked against ZEN's own log lines

`WhiteBalance ID is 52`, `LedWavelength ID is 72`)

| id | key | id | key | id | key |
|---|---|---|---|---|---|
| 0 | IllegalParameterKey | 25 | ColorConversionMode | 50 | TriggerMode |
| 1 | AcquisitionDelay | 26 | **ColorMode** | 51 | ValidPixelMaximum |
| 2 | AcquisitionMode | 27 | ColorSaturation | 52 | WhiteBalance |
| 3 | **AutoExposure** | 28 | ColorSensor | 53 | LightManager |
| 4 | AutoGain | 29 | **ColorTemperature** | 54 | TLIllumination |
| 5 | **AutoWhiteBalance** | 30 | Contrast | 55 | RLIllumination |
| 6 | Binning | 31 | **ExposureGain** | 56 | RLOn |
| 7 | BinningExposureDependency | 32 | **ExposureTime** | 57 | StageXY |
| 8 | BinningList | 33 | FirmwareVersion | 58 | FocusZ |
| 9 | **BlackLevel** | 34 | Frame (ROI) | 59 | TriggerAutoFocus |
| 10 | Brightness | 35 | FrameValidationMode | 60 | CameraAdapter |
| 11 | CameraBias | 36 | **Gamma** | 61 | ReflectorPosition |
| 12 | CameraBitDepth | 37 | **HDRMode** | 62 | ReflectorCount |
| 13 | CameraModel | 38 | MultiChannelMode | 63 | ReflectorMatId |
| 14 | CameraOrientation | 39 | **NoiseCorrection** | 64 | ObjectivePosition |
| 15 | CameraPixelMaximum | 40 | NoiseCorrectionParameters | 65 | ObjectiveCount |
| 16 | CameraPixelMinimum | 41 | ReadoutSpeed | 66 | ObjectiveMatId |
| 17 | CameraPixelDistance | 42 | **Resolution** | 67 | ObjectiveMagnification |
| 18 | CameraPixelType | 43 | Sharpness | 68 | LedCount |
| 19 | CameraSerialNumber | 44 | **SharpnessCorrection** | 69 | LedOn |
| 20 | CameraSize | 45 | TargetTemperature | 70 | LedSelected |
| 21 | CameraTimeout | 46 | Temperature | 71 | LedBrightness |
| 22 | CaptureMode | 47 | TemperatureState | 72 | LedWaveLength |
| 23 | ColorCorrection | 48 | **TransferFormat** | | |
| 24 | ColorCorrectionMatrix | 49 | **TransferQuality** | | |

## Key parameter semantics (Axiocam 208c — from driver + AfterInitialize.xml)

| Key | type | range / values | default | notes |
|---|---|---|---|---|
| ExposureTime | double, **ms** | 0.06–1000 (color; 2000 mono), inc 0.01 | 20.0 | XML: 20 ms |
| ExposureGain | double | 1.0–22.0, inc 1.0 | 1.0 | XML "AnalogGain": 4 |
| AutoExposure | int | 0=off 1=on **2=once** | off | after "once": sleep 200 ms, read back ExposureTime/Gain |
| AutoWhiteBalance | int | 0=off 1=on **2=once** | off | after "once": sleep 200 ms, read back ColorTemperature |
| ColorMode | enum | 0=Mono 1=Color | | XML: 1 |
| ColorTemperature | int | 1500–10000, inc 100 | 5500 | WB presets: 3200K=(0.94722,1,0.48691), 5500K=(0.62657,1,0.55448) |
| WhiteBalance | IndexAndDouble | R,G,B factors | (1,1,1) | READ-ONLY on hardware; software WB = LUT multiply by 1/ratio |
| Gamma | double | 0.1–3.0, inc 0.1 | 1.0 | XML "HardwareGamma": 0.45 |
| BlackLevel | int | read from metadata | 200 | |
| Resolution | enum | 0=3840×2160, 1=1920×1080 | 1 (color) | re-read CameraSize after set! |
| TransferFormat | enum | **0="YUV420" = NV12**, 1=MJPEG | 0 | MJPEG needs hi_mjpeg_dec_w64.dll or cv2.imdecode |
| TransferQuality | int | 1–99 | 90 | MJPEG only |
| CameraBitDepth | int | 8 / 12 | 8 | 12 only for mono |
| CameraSize | IndexAndInteger | [w, h] | read-only | current mode size |
| CameraPixelType | int | 3 = Bgr24 (208c) | | 1≈Gray8, 4≈Bgr48 |
| CameraPixelDistance | IndexAndDouble | [µm_x, µm_y] | 1.85, 1.85 (color) | mono 5.86 |
| CameraTimeout | int | | 5000 | |
| NoiseCorrection / SharpnessCorrection | bool | | true | post-processing |
| Frame | IndexAndInteger | ROI [l,t,w,h] | full sensor | |

## Acquisition semantics (ZEN's own usage)

- **After OpenCamera: sleep 2 s** (ZEN: Thread.Sleep(2000)) — camera settle.
- **Live**: `StartContinuousAcquisition` ONCE with buffer =
  AcquisitionBufferCount (≥5) × single-frame size. Frames arrive via the
  `ImageAcquired` (5) event: EventData {int EventID, pad, long Param1, long Param2}
  where Param1=frame data pointer, Param2=bytecount. Copy Param2 bytes into a
  private buffer, set a "new frame" flag; the acquire loop polls the flag (1 ms sleep).
- **Snap**: `StartSequenceAcquisition(handle, 1, size, buffer)` → poll
  `GetSequenceImage(handle)` (1 ms sleep on ImageNotReady) → on NoError:
  `AbortAcquisition` → decode from buffer.
- After changing ExposureTime: drop 3 live frames (live) / drop `DropFrameCount`
  (default 5) snap frames — stale-frame handling.
- Frame data layout for TransferFormat 0: semi-planar **YUV420** — full-res Y plane
  then ONE interleaved chroma plane (w·h/2 bytes, half-res pairs). HARDWARE-VERIFIED
  pair order is **(V, U)** — first byte of each pair is the red axis, second the blue
  axis. Decoding as U-first (standard NV12) yields a cold blue cast; V-first matches
  ZEN's colors (per-channel correlation 0.84/0.98 vs a ZEN reference frame). ZEN:
  Yuv420SPToYuv420P → IPP YUV420ToRGB24. (Our old decode treated it as planar I420 —
  chroma was garbage; that is why the stream looked gray.)

## Image header

`ApiImage` = 512-byte struct prefixing image data in header-carrying modes:
`{u32 HeaderIdentifier, u16 HeaderVersion, u16 HeaderSize, u16 FrameNumber,
u32 TimeStamp, u16 FrameLeft, u16 FrameTop, u16 FrameWidth, u16 FrameHeight,
u16 PixelType, u16 ValidBits, double ExposureTime}`.
`ApiImageMetadata` = {u64 CaptureFrame, u64 TransferFrame, u64 TimeStamp}.
Actual header size = `ApiInformation.ImageHeaderSize` (via GetLibraryInformation).
The sequence/continuous buffers used by ZEN's live+snap paths carry NO header
(raw NV12 at offset 0).

## Offline findings (saved buffers vs a ZEN reference capture)

- The camera's NV12 stream CONTAINS the full color scene — but captured with
  hardware AWB OFF (cold blue cast; copper stage reads blue).
- ZEN reference = same scene, same debris — vibrant purple/orange (hardware AWB ON).
- 3×4 affine matrix fit raw→ZEN on one scene: mean err 1.8/255 (near-perfect),
  but does NOT generalize across exposure changes (drift) → hardware AWB
  (`AutoWhiteBalance=1` + optionally ColorTemperature) is the real fix; a
  software matrix is only an interim calibration.
- idx2 full-res buffer (5472×3600 + 128 B) format still unresolved — gallery +
  hardware re-capture with explicit pixel-type settings; check
  ApiInformation.ImageHeaderSize + GetImageMetadata on hardware.
