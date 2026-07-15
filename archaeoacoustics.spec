# PyInstaller build spec.
#   pip install pyinstaller
#   pyinstaller archaeoacoustics.spec
# Output: dist/Archaeoacoustics.exe

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=[],
    datas=[("templates", "templates")],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "tkinter",
        "PIL",
        "pytest",
        "IPython",
        "notebook",
        "pandas",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="Archaeoacoustics",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
