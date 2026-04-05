import argparse
import re
import sys
from pathlib import Path

import modal


def collect_files(entry: Path) -> dict[str, str]:
    """Recursively collect a .cu file and all its local #include dependencies."""
    base_dir = entry.parent.resolve()
    files: dict[str, str] = {}

    def _collect(path: Path):
        rel = str(path.resolve().relative_to(base_dir))
        if rel in files:
            return
        content = path.read_text()
        files[rel] = content
        for match in re.finditer(r'#include\s+"([^"]+)"', content):
            dep = (path.parent / match.group(1)).resolve()
            if dep.exists():
                _collect(dep)

    _collect(entry)
    return files


def main():
    parser = argparse.ArgumentParser(
        prog="mcc", description="Compile and Run CUDA C scripts with Modal"
    )
    parser.add_argument("input", help="Input CUDA C file", type=Path)
    parser.add_argument(
        "--app",
        help="App name to run the CUDA script; defaults to the input file name",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        help="Choose the GPU to run the script",
        type=str,
        default="T4",
        choices=[
            "T4",
            "L4",
            "A10",
            "A100",
            "A100-40GB",
            "A100-80GB",
            "L40S",
            "H100",
            "H200",
            "B200",
        ],
    )
    parser.add_argument(
        "--image",
        type=str,
        default="nvidia/cuda:12.4.1-devel-ubuntu22.04",
        help="Container image to use for Modal (registry reference)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60 * 10,
        help="Execution timeout in seconds for the CUDA run (default: 600)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("out"),
        help="Directory to write the compiled binary (default: out/)",
    )
    parser.add_argument(
        "--nvcc-arg",
        dest="nvcc_args",
        action="append",
        default=None,
        metavar="FLAG",
        help="Extra argument to pass to nvcc; repeat for multiple flags",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt and proceed automatically",
    )

    if "--" in sys.argv:
        split = sys.argv.index("--")
        passthrough_nvcc_args = sys.argv[split + 1:]
        sys.argv = sys.argv[:split]
    else:
        passthrough_nvcc_args = []

    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"File {args.input} not found")

    if args.input.suffix.lower() != ".cu":
        parser.error("Input file must have a .cu extension")

    if not args.input.read_text().strip():
        parser.error(f"File {args.input} is empty")

    files = collect_files(args.input)
    entry_rel = str(args.input.resolve().relative_to(args.input.parent.resolve()))

    print(f"Files to be submitted ({len(files)}):")
    for rel in sorted(files):
        print(f"  {rel}")

    if not args.yes:
        answer = input("\nProceed? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    nvcc_args = list(args.nvcc_args or []) + passthrough_nvcc_args

    app_name = args.app or args.input.name
    cuda_image = modal.Image.from_registry(args.image, add_python="3.14")
    app = modal.App(app_name, image=cuda_image)

    @app.function(gpu=args.gpu, timeout=args.timeout, serialized=True)
    def compile_and_run_cuda(files: dict[str, str], entry: str, nvcc_args: list[str] | None = None):
        import subprocess
        import shutil
        import uuid
        from pathlib import Path

        run_id = uuid.uuid4().hex
        work_dir = Path(f"/tmp/{run_id}")
        work_dir.mkdir(parents=True, exist_ok=True)
        binary_path = work_dir / "binary"

        for rel_path, content in files.items():
            dest = work_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)

        def _run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
            print(f"$ {' '.join(cmd)}")

            completed = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if completed.stdout:
                print(completed.stdout)

            if completed.stderr:
                print(completed.stderr)

            return completed

        try:
            extra_args = nvcc_args or []

            # Debug step: dump PTX (no ptxas) so we can inspect generated instructions
            ptx_path = work_dir / "debug.ptx"
            _run_command(
                ["nvcc", str(work_dir / entry), "--ptx", "-o", str(ptx_path), *extra_args]
            )
            if ptx_path.exists():
                ptx_text = ptx_path.read_text()
                # Print lines containing tcgen05 instructions for debugging
                tcgen05_lines = [
                    f"  {i+1}: {line}"
                    for i, line in enumerate(ptx_text.splitlines())
                    if "tcgen05" in line
                ]
                if tcgen05_lines:
                    print("=== tcgen05 PTX instructions (first 40) ===")
                    for l in tcgen05_lines[:40]:
                        print(l)
                    print(f"=== ({len(tcgen05_lines)} total tcgen05 lines) ===")

            compile_result = _run_command(
                ["nvcc", str(work_dir / entry), "-o", str(binary_path), *extra_args]
            )

            if compile_result.returncode != 0:
                raise RuntimeError("nvcc failed to compile the CUDA source file")

            execution_result = _run_command([str(binary_path)])

            if execution_result.returncode != 0:
                raise RuntimeError(
                    "Executable exited with a non-zero status while running the CUDA program"
                )

            return execution_result.stdout, binary_path.read_bytes()

        finally:
            if work_dir.exists():
                shutil.rmtree(work_dir)

    out_dir = args.out
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / args.input.stem

    with modal.enable_output():
        with app.run():
            _, binary = compile_and_run_cuda.remote(files, entry_rel, nvcc_args)

    out_path.write_bytes(binary)
    print(f"Binary saved to {out_path}")


if __name__ == "__main__":
    main()
