import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys


class CSimResult(ctypes.Structure):
    _fields_ = [
        ("loss", ctypes.c_double),
        ("wait_avg", ctypes.c_double),
        ("wait_max", ctypes.c_double),
        ("urllc_tot", ctypes.c_double),
        ("urllc_max", ctypes.c_double),
        ("embb_tot", ctypes.c_double),
    ]


dir = os.path.dirname(os.path.realpath(__file__))
c_file = Path(os.path.join(dir, "simulation.c"))

if sys.platform.startswith("win"):
    lib_file = Path(os.path.join(dir, "simulation.dll"))
elif sys.platform == "darwin":
    lib_file = Path(os.path.join(dir, "simulation.dylib"))
else:
    lib_file = Path(os.path.join(dir, "simulation.so"))

if not c_file.exists():
    raise FileNotFoundError("Simulation C code does not exist.")

if sys.platform.startswith("win"):
    gcc = shutil.which("gcc")
    if gcc:
        compile_command = [
            gcc,
            "-shared",
            "-o",
            str(lib_file),
            str(c_file),
        ]
    else:
        cl = shutil.which("cl")
        if not cl:
            raise EnvironmentError(
                "No C compiler found. Install MinGW-w64 (gcc) or MSVC (cl)."
            )
        compile_command = [
            cl,
            "/LD",
            str(c_file),
            "/link",
            "/OUT:" + str(lib_file),
        ]
else:
    gcc = shutil.which("gcc")
    if not gcc:
        raise EnvironmentError("No C compiler found. Please install gcc.")
    compile_command = [
        gcc,
        "-shared",
        "-o",
        str(lib_file),
        "-fPIC",
        str(c_file),
        "-lm",
        "-lpthread",
    ]
try:
    subprocess.run(compile_command, check=True, capture_output=True, text=True)
except subprocess.CalledProcessError as e:
    print("C compilation failed!")
    print(e.stderr)
    raise

# Load the shared library
C_LIB = ctypes.CDLL(str(lib_file))

# Define the function signature
C_LIB.simu.argtypes = [
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(CSimResult),
]
C_LIB.simu.restype = None
