# App icons

Drop the icon files here and the app picks them up automatically
(no code changes needed):

- **`talos.ico`** — the Windows icon, used for the app windows, the
  taskbar, and the frozen EXE (PyInstaller embeds it). Multi-resolution
  ICO recommended: 16 / 24 / 32 / 48 / 64 / 128 / 256 px.
- **`talos.png`** — optional fallback (also handy for non-Windows
  platforms); a 256 px PNG works everywhere.

Precedence: `talos.ico` wins when both are present. The app runs fine
with neither (it falls back to the default window icon).

> The `talos.png` currently in this folder is a **placeholder**; the real
> icon is still being designed. Replace it (and add the matching `.ico`)
> when it is ready.

After dropping a new icon, rebuild the frozen app to embed it into the
EXE (`packaging/talos.spec` bundles this folder and uses the .ico as the
executable icon when it exists).
