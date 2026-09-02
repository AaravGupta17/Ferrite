# Installing Ferrite

## Bottom Line Up Front

Clone the repo, install a toolchain (GCC/Clang + CMake + Ninja) for your OS, then configure and build. There are no third-party libraries to fetch — after the toolchain, `cmake -S . -B build && cmake --build build` is the whole story.

## 1. Get the source

```sh
git clone https://github.com/AaravGupta17/Ferrite.git
cd Ferrite
```

Or download and extract the ZIP if you don't use Git.

## 2. Install the toolchain

### Windows — Scoop (recommended, simple)

Open PowerShell and install [Scoop](https://scoop.sh) if you don't have it:

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex
```

Then install the three tools:

```powershell
scoop install main/gcc
scoop install main/cmake
scoop install main/ninja
```

After installing, open a **new** terminal so the updated PATH takes effect.

> **Sanitizers.** Scoop's `gcc` often lacks the ASan/UBSan runtimes. Ferrite still builds and tests fine without them — it just skips the memory checks. If you want sanitizers on Windows, use MSYS2 instead (next option).

### Windows — MSYS2 (best if you want ASan/UBSan)

1. Download and install [MSYS2](https://www.msys2.org).
2. Open the **MSYS2 UCRT64** terminal and install the toolchain:

```sh
pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja
```

3. Add MSYS2's UCRT64 `bin` to your PATH, or run the CMake commands from inside the MSYS2 terminal.

### Linux / WSL

```sh
# Debian / Ubuntu
sudo apt update && sudo apt install -y build-essential cmake ninja-build

# Fedora
sudo dnf install -y gcc gcc-c++ cmake ninja-build
```

### macOS

Install the Xcode Command Line Tools (provides Clang), then Homebrew for CMake and Ninja:

```sh
xcode-select --install
brew install cmake ninja
```

## 3. Verify

```sh
gcc --version    # or: clang --version
cmake --version
ninja --version
```

All three should print version numbers. If you installed on Windows and the commands aren't found, close and reopen your terminal so PATH refreshes.

## 4. Configure and build

From the repo root:

```sh
cmake -S . -B build -G Ninja
cmake --build build
```

`cmake --build build` compiles all twelve test binaries plus the subsystem libraries. The demo and benchmarks are opt-in — pass `--target <name>` to build them:

```sh
cmake --build build --target demo
cmake --build build --target bench_avx2
```

## 5. Run the tests

```sh
ctest --test-dir build
```

You should see `100% tests passed out of 12`.

## 6. Next steps

- **Run the demo:** `cmake --build build --target demo && ./build/demo`
- **Run the benchmarks:** see `documentation/README.md#5-run-the-benchmarks`
- **Understand the codebase:** see `documentation/README.md#layout-at-a-glance`

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Ferrite requires GCC or Clang` at configure | You're using MSVC or another unsupported compiler. Install GCC/Clang and clear the `build/` cache. |
| `ASan/UBSan requested but unavailable` | Not an error. Build without sanitizers (Scoop MinGW often hits this). Install MSYS2 to get them. |
| `libwinpthread-1.dll` missing when running an exe | Rebuild that target — it copies the DLL next to `build/*.exe` automatically. Don't move exes outside `build/`. |
| Tests referencing `tests/tiny_mlp.onnx` fail to find the file | Tests must run from the repo root (CTest handles this). Don't run test exes from another cwd. |
