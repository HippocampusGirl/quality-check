from pathlib import Path


def split_ext(path: str | Path) -> tuple[str, str]:
    """Splits filename and extension (.gz safe)
    >>> split_ext('some/file.nii.gz')
    ('file', '.nii.gz')
    >>> split_ext('some/other/file.nii')
    ('file', '.nii')
    >>> split_ext('otherext.tar.gz')
    ('otherext', '.tar.gz')
    >>> split_ext('text.txt')
    ('text', '.txt')

    Adapted from niworkflows
    """
    from pathlib import Path

    if isinstance(path, str):
        path = Path(path)

    name = str(path.name)

    safe_name = name
    for compound_extension in [".gz", ".xz"]:
        safe_name = safe_name.removesuffix(compound_extension)

    stem = Path(safe_name).stem
    return stem, name[len(stem) :]
