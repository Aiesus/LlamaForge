# PyInstaller spec for llama-gui v2
# Build: pyinstaller build.spec
#
# Produces a single-directory bundle at dist/llama-gui/
# tool_proxy.py is included as a data file so setup wizard can deploy it to WSL.

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[str(Path('.').resolve())],
    binaries=[],
    datas=[
        # Bundle tool_proxy.py so the setup wizard can copy it to WSL
        ('tool_proxy.py', '.'),
    ],
    hiddenimports=[
        'tkinter',
        'tkinter.ttk',
        'tkinter.scrolledtext',
        'tkinter.messagebox',
        'tkinter.simpledialog',
        'tkinter.filedialog',
        'urllib.request',
        'json',
        'threading',
        'subprocess',
        'pathlib',
        'webbrowser',
        'csv',
        'io',
        'itertools',
        'dataclasses',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'aiohttp',          # only needed inside WSL for tool_proxy.py
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='llama-gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # no console window; logging goes to the GUI log panel
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='llama-gui',
)
