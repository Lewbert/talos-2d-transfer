# Third-party notices

TALOS itself is licensed under the **GNU General Public License, version 3
(only)** — see [LICENSE](LICENSE). It is built on the components below; their
licences apply to them. The full licence texts travel with the build
(`licenses/` beside `talos.exe`) and live in this repository under
[`packaging/licenses/`](packaging/licenses/).

## In the application and its build

| Component | Licence | Text |
|---|---|---|
| Python 3.12 (`python312.dll`) | PSF-2.0 | [`Python-LICENSE.txt`](packaging/licenses/Python-LICENSE.txt) |
| PySide6 / shiboken6 | LGPL-3.0-only — used under the LGPL option; The Qt Company also offers GPL and commercial options, and the wheels' own `licenses/LicenseRef-Qt-Commercial.txt` names the last | [`LGPL-3.0-only.txt`](packaging/licenses/LGPL-3.0-only.txt) |
| Qt 6 (Core, Gui, Widgets and the Qt plugins PyInstaller collects) | LGPL-3.0-only | [`LGPL-3.0-only.txt`](packaging/licenses/LGPL-3.0-only.txt) |
| numpy | BSD-3-Clause, plus 0BSD / MIT / Zlib / CC0-1.0 components | [`numpy-LICENSE.txt`](packaging/licenses/numpy-LICENSE.txt), with the per-module texts under [`numpy-bundled/`](packaging/licenses/numpy-bundled/) |
| OpenCV, distributed as `opencv-python` | Apache-2.0 (the Python packaging around it is MIT) | [`opencv-python-LICENSE.txt`](packaging/licenses/opencv-python-LICENSE.txt) |
| FFmpeg — bundled inside `cv2/` by `opencv-python` | LGPL-2.1-or-later | the FFmpeg section of [`opencv-python-LICENSE-3RD-PARTY.txt`](packaging/licenses/opencv-python-LICENSE-3RD-PARTY.txt) |
| pyserial | BSD-3-Clause | [`pyserial-LICENSE.txt`](packaging/licenses/pyserial-LICENSE.txt) |
| PyInstaller bootloader (embedded in `talos.exe`) | GPL-2.0-or-later with the special exception that permits distributing bundled applications | [`PyInstaller-COPYING.txt`](packaging/licenses/PyInstaller-COPYING.txt) |

### Qt, under the LGPL

The Qt libraries ship as separate, dynamically linked DLLs (`Qt6Core.dll`,
`Qt6Gui.dll`, `Qt6Widgets.dll` and the platform/image plugins) beside
`talos.exe` in the one-dir build, so you may replace them with modified
versions. The source they are built from is available from
<https://download.qt.io/> and from the PySide6 wheels on PyPI; nothing in
TALOS links Qt statically.

## Development and build only

- **pytest** — MIT.
- **PyInstaller** — GPL-2.0-or-later with the bundling exception (above); it
  is a build tool, and only its bootloader ships inside `talos.exe`.
- The optional, *unwired* camera backends — **harvesters** (Apache-2.0),
  **pymmcore-plus** (BSD-3-Clause) and **genicam** (the GenICam licence,
  `LicenseRef-GenICam-1.6`) — are excluded from the frozen build.
