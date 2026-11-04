"""The data store: where a dataset comes from, in one of two modes chosen per dataset.

fetch: public data that is easy to download (torchvision, Hugging Face, an open link). The session downloads it onto the
       machine's own disk and nothing is kept in the artifacts folder, so it costs no Drive space.
store: data that needs a login, or that was kept on purpose. It sits once per project in <artifacts>/data/<name>/ (or as
       <name>.zip), synced with Drive, with a source.md note recording its origin and a checksum for every file. Each
       session checks the data against that note, copies it onto the machine's disk and unpacks a zip there, because the
       disk reads far faster than Drive.

A run never fills the data folder by itself, and credentials are never stored: a dataset behind a login is downloaded by
the person who holds the login, and keep_copy() is used only after that person says to keep a fetched dataset.
"""
import datetime
import hashlib
import re
import shutil
import stat
import tarfile
import warnings
import zipfile
from pathlib import Path, PurePosixPath

ARCHIVE_SUFFIXES = (".zip", ".tar.gz", ".tgz", ".tar")
MODES = ("fetch", "store")
MANIFEST_NAME = "source.md"
READY_MARKER = ".unpacked"
FETCHED_MARKER = ".fetched"
_MARKERS = (MANIFEST_NAME, READY_MARKER, FETCHED_MARKER)
# Files that Windows, macOS or a sync client add to a folder on their own; they are never dataset content, so a kept dataset is not
# reported as having an unlisted file because of one (compared in lower case, at any depth).
_SYSTEM_FILES = frozenset({"desktop.ini", ".ds_store", "thumbs.db"})

# The one format of a file line in a source note; the dataset checker that runs before a run reads exactly the same format:
# "- <relative path>", exactly two spaces, "sha256=", then 64 lowercase hexadecimal digits, and nothing after.
_FILE_LINE = re.compile(r"- (?P<name>\S(?:.*\S)?)  sha256=(?P<digest>[0-9a-f]{64})")
_LINE_FORMAT = "'- <path>  sha256=<64 lowercase hex digits>' with exactly two spaces before sha256"
_MARKER_DIGEST = re.compile(r"sha256=([0-9a-f]{64})")
_UNSAFE_NAME = ("{manifest} lists {name!r}, which is not a plain relative path inside the dataset (no absolute path, drive letter "
                "or colon, backslash, '..' or empty part, or space at either end), so it was not read")


class DatasetMissing(RuntimeError):
    """Raised when a dataset is not where its mode says; the message says how to add it."""


class ChecksumMismatch(RuntimeError):
    """Raised when a kept dataset does not match its source note or the expected checksum; the message names the file."""


def sha256_of(path):
    """Returns the hex SHA-256 of a file, read in 1 MB pieces so large archives do not fill memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for piece in iter(lambda: handle.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def find_archive(name, data_root):
    """Returns the archive file for dataset `name` in data_root, or None."""
    for suffix in ARCHIVE_SUFFIXES:
        candidate = Path(data_root) / f"{name}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _inside(target, member_name):
    """Checks that an archive member lands inside `target`; raises ValueError when it would escape."""
    destination = (target / member_name).resolve()
    if target.resolve() not in destination.parents and destination != target.resolve():
        raise ValueError(f"archive member {member_name!r} would be written outside the target folder")


def _vet_zip(target, bundle):
    """Raises ValueError for a zip entry that escapes target, is a symbolic link, or is any other special file.

    Zip keeps the Unix file type in the high 16 bits of external_attr; 0 means the archive recorded no type.
    """
    for info in bundle.infolist():
        _inside(target, info.filename)
        kind = stat.S_IFMT(info.external_attr >> 16)
        if kind == stat.S_IFLNK:
            raise ValueError(f"archive member {info.filename!r} is a link, which is not allowed")
        if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError(f"archive member {info.filename!r} is a special file, not a regular file or folder")


def _vet_tar(target, bundle):
    """Raises ValueError for a tar member that escapes target, is a link, or is not a regular file or folder (FIFO, device)."""
    for member in bundle.getmembers():
        _inside(target, member.name)
        if member.issym() or member.islnk():
            raise ValueError(f"archive member {member.name!r} is a link, which is not allowed")
        if not (member.isreg() or member.isdir()):
            raise ValueError(f"archive member {member.name!r} is a special file, not a regular file or folder")


def _fresh_folder(target):
    """Makes `target` an empty folder, removing whatever an earlier unpack (or a failed one) left there."""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)


def _unpack(archive, target):
    """Unpacks a zip or tar archive into an empty `target`, refusing any member that escapes it, is a link or is special.

    Every member is checked before anything is written, so a refused archive leaves nothing behind.
    """
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            _vet_zip(target, bundle)
            _fresh_folder(target)
            bundle.extractall(target)
    else:
        with tarfile.open(archive) as bundle:
            _vet_tar(target, bundle)
            _fresh_folder(target)
            if hasattr(tarfile, "data_filter"):           # Python 3.12+ (and security releases): the safe "data" filter
                bundle.extractall(target, filter="data")
            else:
                bundle.extractall(target)


def _unpacked_digest(target):
    """Returns the archive checksum recorded in target's .unpacked marker, or None when there is no usable marker."""
    try:
        found = _MARKER_DIGEST.search((Path(target) / READY_MARKER).read_text(encoding="utf-8"))
    except OSError:
        return None
    return found.group(1) if found else None


def _check_name(name):
    """Raises ValueError unless `name` is one plain folder name (it is used to build, and to delete, a work folder).

    Refused: empty, "." and "..", a slash, backslash or colon (separators and drive letters), NUL, and whitespace at either
    end. The dataset checker that runs before a run applies the same rule to the name in the specification.
    """
    if name in ("", ".", "..") or name != name.strip() or re.search(r"[\\/:\x00]", name):
        raise ValueError(f"dataset name {name!r} must be one plain folder name")


def _plain_relative(name):
    """Tells whether a file name listed in a source note is a plain relative path that stays inside the dataset.

    False for an absolute path, a drive letter or any colon, a backslash, NUL, a ".." part, an empty part (a leading,
    trailing or doubled slash), and whitespace at either end of the name or of any part. The same rule is in the dataset checker.
    """
    if name != name.strip() or re.search(r"[\\:\x00]", name):
        return False
    return all(part and part != ".." and part == part.strip() for part in name.split("/"))


def _escapes(base, path):
    """Tells whether `path`, after following links, lies outside the folder `base`."""
    root, resolved = Path(base).resolve(), Path(path).resolve()
    return resolved != root and root not in resolved.parents


def _verify_against_note(target, digests=None, warn_if_missing=True):
    """Raises ChecksumMismatch when a dataset that has a source note does not match it.

    A dataset with no note still passes, but it was not checked, so this warns (UserWarning) unless warn_if_missing is
    False. `digests` maps a file name to a checksum already computed, so a large archive is not read twice.
    """
    manifest, _, _ = _manifest_target(target)
    if manifest.is_file():
        problems = check_manifest(target, digests)
        if problems:
            raise ChecksumMismatch(f"{Path(target).name} does not match its source note {manifest.name}: " + "; ".join(problems))
    elif warn_if_missing:
        warnings.warn(f"{Path(target).name} has no source note ({manifest.name}), so its files were loaded without a checksum "
                      f"check. Write one with datastore.write_manifest.", UserWarning, stacklevel=3)


def ensure_local_copy(name, data_root, work_dir, expected_sha256=None):
    """Returns a readable folder for a kept dataset, unpacking an archive into work_dir when there is one.

    A kept dataset that has a source note is checked against it on every call: a changed, missing or unlisted file raises
    ChecksumMismatch naming the file. A dataset with no note still loads (datacheck warns about the missing note), and
    expected_sha256, when given, is checked against the archive as well. An archive is unpacked into a clean
    work_dir/<name> again whenever its checksum differs from the one the .unpacked marker recorded.

    Raises DatasetMissing when it is not there (it never downloads), ChecksumMismatch for a wrong archive or folder,
    and ValueError for an archive that would write outside work_dir or holds a link or special file.
    """
    _check_name(name)
    data_root, work_dir = Path(data_root), Path(work_dir)
    archive = find_archive(name, data_root)
    if archive is not None:
        digest = sha256_of(archive)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ChecksumMismatch(f"{archive.name} does not match the expected checksum")
        _verify_against_note(archive, {archive.name: digest}, warn_if_missing=expected_sha256 is None)
        target = work_dir / name
        if _unpacked_digest(target) != digest:
            _unpack(archive, target)
            (target / READY_MARKER).write_text(f"unpacked sha256={digest}\n", encoding="utf-8")
        return target
    folder = data_root / name
    if folder.is_dir() and any(folder.iterdir()):
        _verify_against_note(folder)
        return folder
    raise DatasetMissing(
        f"dataset {name!r} is set to store mode but is not in {data_root}. Put it there once (a person downloads it if it "
        f"needs a login, or keep_copy() saves a fetched one); it is never downloaded during a run in store mode.")


def resolve(name, mode, data_root, work_dir, fetch=None, expected_sha256=None):
    """Returns a readable folder for dataset `name` under its mode.

    store: the kept copy in data_root, verified against its source note when it has one (an archive is unpacked into
           work_dir); never calls fetch.
    fetch: fetch(folder) downloads into work_dir/name once; data_root is never written.
    """
    if mode not in MODES:
        raise ValueError(f"data mode {mode!r} must be one of {MODES}")
    if mode == "store":
        return ensure_local_copy(name, data_root, work_dir, expected_sha256)
    if fetch is None:
        raise ValueError("fetch mode needs a fetch function")
    target = Path(work_dir) / name
    if not (target / FETCHED_MARKER).exists():
        fetch(target)
        target.mkdir(parents=True, exist_ok=True)
        (target / FETCHED_MARKER).write_text("fetched\n")
    return target


def _files_under(folder):
    """Returns the dataset's files as sorted relative POSIX names, leaving out the manifest and the markers."""
    names = [p.relative_to(folder).as_posix() for p in Path(folder).rglob("*") if p.is_file()]
    return sorted(n for n in names if n not in _MARKERS and n.rsplit("/", 1)[-1].lower() not in _SYSTEM_FILES)


def _manifest_target(target):
    """Returns (manifest path, base folder, file names) for a dataset folder or a single archive file."""
    target = Path(target)
    if target.is_dir():
        return target / MANIFEST_NAME, target, _files_under(target)
    stem = target.name
    for suffix in ARCHIVE_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return target.with_name(f"{stem}.{MANIFEST_NAME}"), target.parent, [target.name]


def write_manifest(target, source, licence, acquired_by, credentials_needed=False, today=None):
    """Writes the source note for a dataset folder (<folder>/source.md) or an archive (<name>.source.md).

    It records the origin, licence, date, who acquired it and a SHA-256 for every file. Never put a username, password
    or token in `source`; say only that credentials were needed.
    """
    manifest, base, names = _manifest_target(target)
    day = today or datetime.date.today().isoformat()
    login = "yes (held by the person who downloaded it, never stored)" if credentials_needed else "no"
    lines = [f"# {Path(target).name}", "", f"- source: {source}", f"- licence: {licence}", f"- acquired: {day} by {acquired_by}",
             f"- credentials needed: {login}", "", "## Files", ""]
    lines += [f"- {name}  sha256={sha256_of(base / name)}" for name in names]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_note(manifest):
    """Returns (entries, problems) from a source note: (name, digest) for each valid file line, and a sentence for each malformed one.

    A line that starts with "- " and contains "sha256=" must match the one file-line format; any other line is prose. A
    line in the right format whose name is not a plain relative path is a problem too, and is not returned as an entry.
    """
    entries, problems = [], []
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        match = _FILE_LINE.fullmatch(line)
        if match and not _plain_relative(match["name"]):
            problems.append(_UNSAFE_NAME.format(manifest=manifest.name, name=match["name"]))
        elif match:
            entries.append((match["name"], match["digest"]))
        elif line.startswith("- ") and "sha256=" in line:
            problems.append(f"line {number} of {manifest.name} ({line!r}) is not in the form {_LINE_FORMAT}, so it checks nothing")
    return entries, problems


def check_manifest(target, digests=None):
    """Returns a list of problems (empty when every recorded file is present and unchanged and every dataset file is listed).

    Problems: no source note; a malformed file line; a note that lists no files (it would verify nothing); a listed file
    that is missing or has changed; a file of the dataset that the note does not list. `digests` optionally maps a file
    name to a checksum already computed, to avoid reading a large file twice.
    """
    manifest, base, names = _manifest_target(target)
    if not manifest.is_file():
        return [f"{manifest.name} is missing"]
    entries, problems = _read_note(manifest)
    if not entries:
        problems.append(f"{manifest.name} lists no files, so it verifies nothing")
    known = digests or {}
    listed = set()
    for name, digest in entries:
        path = base / name
        if _escapes(base, path):                  # a link inside the dataset that leads out of it
            problems.append(_UNSAFE_NAME.format(manifest=manifest.name, name=name))
            continue
        listed.add(PurePosixPath(name).as_posix())
        if not path.is_file():
            problems.append(f"{name} is missing")
        elif (known.get(name) or sha256_of(path)) != digest:
            problems.append(f"{name} has changed since it was recorded")
    problems += [f"{name} is in the dataset but not listed in {manifest.name}"
                 for name in names if name not in listed and (base / name).is_file()]
    return problems


def keep_copy(name, work_dir, data_root, source, licence, acquired_by, credentials_needed=False, archive=False, today=None):
    """Keeps a fetched dataset by copying it into the data folder, with its source note; use only when a person says to.

    With archive=True the copy is one zip file (smaller to store). Returns the new folder or zip path. Raises
    DatasetMissing when the dataset was never fetched and FileExistsError when anything is already kept under the name:
    a folder, a file, or an archive of any supported kind.
    """
    source_dir = Path(work_dir) / name
    if not source_dir.is_dir():
        raise DatasetMissing(f"dataset {name!r} has not been fetched into {work_dir}, so there is nothing to keep")
    data_root = Path(data_root)
    destination = data_root / (f"{name}.zip" if archive else name)
    if (data_root / name).exists() or any((data_root / f"{name}{suffix}").exists() for suffix in ARCHIVE_SUFFIXES):
        raise FileExistsError(f"a kept copy of {name!r} already exists in {data_root}")
    data_root.mkdir(parents=True, exist_ok=True)
    if archive:
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as bundle:
            for relative in _files_under(source_dir):
                bundle.write(source_dir / relative, relative)
    else:
        shutil.copytree(source_dir, destination, ignore=shutil.ignore_patterns(*_MARKERS))
    write_manifest(destination, source, licence, acquired_by, credentials_needed, today)
    return destination
