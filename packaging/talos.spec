# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for TALOS (one-dir build, no self-zipping).

SmartCamApi only: the dead backends (harvesters/GenTL, pymmcore_plus
Micro-Manager, MCam) are unwired from the app — their optional deps are
EXCLUDED from the frozen build (the registry registers them lazily and
skips them when the imports fail), which keeps the bundle lean.
"""

import os

ROOT = os.path.dirname(SPECPATH)  # repo root — relative paths resolve here

datas = [
    (os.path.join(ROOT, "resources", "defaults", "default_settings.json"),
     "resources/defaults"),
]
# The QSS image glyphs (spin-button arrows, the checkbox dot, the combo
# chevron). Without them the frozen app silently loses those affordances:
# Qt's stylesheet loads them at runtime by path, and a missing image is a
# warning, not an error.
qss_dir = os.path.join(ROOT, "resources", "qss")
if os.path.isdir(qss_dir):
    datas.append((qss_dir, "resources/qss"))
# the app icons (talos.ico / talos.png) — bundled once the user drops
# them into resources/icons/; the EXE icon comes from the same file
icons_dir = os.path.join(ROOT, "resources", "icons")
icon_path = os.path.join(icons_dir, "talos.ico")
if os.path.isdir(icons_dir):
    datas.append((icons_dir, "resources/icons"))
# The licences: TALOS's own GPL text, the third-party notices, and the
# components' texts under packaging/licenses/ — the LGPL-3.0 text among them
# is required to travel with the Qt DLLs this build ships.
for _lic in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
    _lic_path = os.path.join(ROOT, _lic)
    if os.path.isfile(_lic_path):
        datas.append((_lic_path, "licenses"))
_licenses_dir = os.path.join(ROOT, "packaging", "licenses")
if os.path.isdir(_licenses_dir):
    datas.append((_licenses_dir, "licenses"))

hiddenimports = [
    "serial",
]

a = Analysis(
    [os.path.join(ROOT, "talos", "__main__.py")],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[os.path.join(ROOT, "packaging", "hooks")],
    excludes=[
        "tkinter",
        "matplotlib",
        "pytest",
        "pymmcore_plus",
        "harvesters",
        "genicam",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtQml",
        "PySide6.Qt3DCore",
        "PySide6.QtCharts",
        "PySide6.QtQuick",
        "PySide6.QtNetworkAuth",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe_args = dict(
    name="talos",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)
if os.path.exists(icon_path):
    exe_args["icon"] = icon_path

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    **exe_args,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="TALOS",
)
