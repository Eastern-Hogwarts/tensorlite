import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}

# ROOT/python/setup.py -> ROOT
PROJECT_ROOT = Path(__file__).parent.resolve()
PROJECT_BUILD_DIR = (PROJECT_ROOT / Path("build")).resolve()
PACKAGE_NAME = "pytensorlite"

def process_submodules(submodule_config_path):
    pattern = r"path\s*=\s*(.+)\n\s*url\s*=\s*(.+)\n"
    with open(submodule_config_path, 'r') as f:
        contents = f.read()

    matches = re.findall(pattern, contents)
    for path, url in matches:
        if os.path.exists(os.path.join(PROJECT_ROOT, path)):
            continue
        else:
            subprocess.run(
                f"git clone --depth=1 {url} {path}".split(), check=True
            )

# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = PROJECT_ROOT / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()
        cdlls_output_path = f"{extdir}{os.sep}{PACKAGE_NAME}{os.sep}"

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={cdlls_output_path}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={cdlls_output_path}",
            f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={cdlls_output_path}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DBUILD_PYTHON_API=ON",
            f"-DBUILD_TEST=OFF",
        ]
        build_args = []

        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja  # noqa: F401

                    ninja_exec_path = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_exec_path}",
                    ]

                except ImportError:
                    pass
        else:
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={cdlls_output_path}",
                    f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{cfg.upper()}={cdlls_output_path}",
                    f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{cfg.upper()}={cdlls_output_path}",
                ]
                build_args += ["--config", cfg]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        # if we are under sdist dir, fetch all submodules manually.
        process_submodules(os.path.join(PROJECT_ROOT, ".gitmodules"))

        subprocess.run(
            ["cmake", f"-S{ext.sourcedir}", f"-B{PROJECT_BUILD_DIR}"] + cmake_args, check=True
        )

        subprocess.run(
            ["cmake", "--build", f"{PROJECT_BUILD_DIR}"] + build_args, check=True
        )

# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name=PACKAGE_NAME,
    version="0.1",
    ext_modules=[CMakeExtension(f"{PACKAGE_NAME}_capi", sourcedir=PROJECT_ROOT)],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.7",
    package_dir={"": "python"},
    packages=find_packages(
        where="python",
        include=["pytensorlite"],
    ),
    include_package_data=True
)
