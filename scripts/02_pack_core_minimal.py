"""Minimal core packaging script to bundle only essential code and config."""

from pathlib import Path
import zipfile


def pack_core_minimal(output_filename: str = "smart_scope_core_minimal.zip") -> None:
    root = Path(__file__).resolve().parent.parent
    output_path = root / output_filename

    # Minimal core folders
    include_folders = [
        "aiming_engine",
        "bridge",
        "detector+tracker",
        "config",
    ]

    # Minimal core files
    include_files = [
        "requirements.txt",
    ]

    print(f"Creating minimal core package: {output_path.name}")
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for folder_name in include_folders:
            folder_path = root / folder_name
            if folder_path.exists():
                for file_path in folder_path.rglob("*"):
                    if file_path.is_file() and "__pycache__" not in file_path.parts:
                        rel_path = file_path.relative_to(root)
                        zipf.write(file_path, rel_path)
                        print(f"Added core: {rel_path}")

        for file_name in include_files:
            file_path = root / file_name
            if file_path.exists() and file_path.is_file():
                zipf.write(file_path, file_name)
                print(f"Added core file: {file_name}")

    print(f"\nMinimal core package created successfully: {output_path}")


if __name__ == "__main__":
    pack_core_minimal()
