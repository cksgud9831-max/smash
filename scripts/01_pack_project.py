"""Packaging script to bundle project files for team sharing."""

from pathlib import Path
import zipfile


def pack_project(output_filename: str = "smart_scope_project.zip") -> None:
    root = Path(__file__).resolve().parent.parent
    output_path = root / output_filename

    # Folders to include in the deployment package
    include_folders = [
        "aiming_engine",
        "bridge",
        "detector+tracker",
        "config",
        "examples",
        "tests",
        "docs/interim_reports",
        "docs/pdfs",
    ]

    # Individual files to include
    include_files = [
        "requirements.txt",
        "requirements-bridge-hardware.txt",
        "pytest.ini",
        "docs/analysis/known_issues.md",
    ]

    print(f"Creating package: {output_path.name}")
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for folder_name in include_folders:
            folder_path = root / folder_name
            if folder_path.exists():
                for file_path in folder_path.rglob("*"):
                    if file_path.is_file() and "__pycache__" not in file_path.parts:
                        rel_path = file_path.relative_to(root)
                        zipf.write(file_path, rel_path)
                        print(f"Added: {rel_path}")

        for file_name in include_files:
            file_path = root / file_name
            if file_path.exists() and file_path.is_file():
                zipf.write(file_path, file_name)
                print(f"Added: {file_name}")

    print(f"\nPackage created successfully: {output_path}")


if __name__ == "__main__":
    pack_project()
