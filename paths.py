"""Where things live: one function per folder, so code, notebooks and tools never build paths by hand.

Three places hold one project: the vault (notes), this code folder (synced with GitHub) and the artifacts folder
(synced with Google Drive). Artifacts are everything that is not code: data/, scratch/ and exp-NNN/run-NNN/.
On this computer that folder is C:\\Users\\<you>\\ResearchArtifacts\\<project>; on Colab it is the same folder as
seen through the mounted Drive.
"""
import os
import re
from pathlib import Path

# Name of this project; the code folder, the artifacts folder and the Drive folder all use it.
PROJECT = "face-detection-classification"

# The parent folder of all projects' artifacts, on this computer and on Drive.
ARTIFACTS_PARENT = "ResearchArtifacts"

# Where the mounted Drive appears on a Colab machine.
DRIVE_MOUNT = Path("/content/drive")

# Experiment folders are exp-NNN or the throwaway "scratch"; run folders are run-NNN or smoke-NNN.
EXP_PATTERN = re.compile(r"^(exp-\d{3}|scratch)$")
RUN_PATTERN = re.compile(r"^(run|smoke)-\d{3}$")


def project_root():
    """Returns the folder that holds this file: the code folder on this computer, the git clone on Colab."""
    return Path(__file__).resolve().parent


def drive_artifacts_dir(environ=None, mount=DRIVE_MOUNT):
    """Finds this project's artifacts folder on the mounted Drive, or returns None.

    Order: the DRIVE_ARTIFACTS_DIR environment variable; then, for a computer folder synced by the Drive desktop app,
    Othercomputers/<computer name>/ResearchArtifacts/<project> (computer names in alphabetical order); then
    MyDrive/ResearchArtifacts/<project>. The synced computer folder comes first because that is where this computer's
    own copy of the artifacts lives, and the stage scripts and the status probe look there first too.
    """
    environ = os.environ if environ is None else environ
    if environ.get("DRIVE_ARTIFACTS_DIR"):
        return Path(environ["DRIVE_ARTIFACTS_DIR"])
    candidates = sorted((Path(mount) / "Othercomputers").glob(f"*/{ARTIFACTS_PARENT}/{PROJECT}"))
    candidates.append(Path(mount) / "MyDrive" / ARTIFACTS_PARENT / PROJECT)
    return next((c for c in candidates if c.is_dir()), None)


def artifact_root(environ=None, on_colab=None, mount=DRIVE_MOUNT, home=None):
    """Returns the artifacts folder for this machine.

    Order: the ARTIFACT_ROOT environment variable; on Colab the Drive artifacts folder when Drive is mounted,
    otherwise /content/artifacts (lost when the machine ends); on this computer <home>/ResearchArtifacts/<project>
    when that folder exists, otherwise <project>/artifacts. The arguments exist so tests can set the situation.
    """
    environ = os.environ if environ is None else environ
    if environ.get("ARTIFACT_ROOT"):
        return Path(environ["ARTIFACT_ROOT"])
    on_colab = os.path.isdir("/content") if on_colab is None else on_colab
    if on_colab:
        synced = drive_artifacts_dir(environ, mount)
        return synced if synced else Path("/content/artifacts")
    shared = Path(home or Path.home()) / ARTIFACTS_PARENT / PROJECT
    return shared if shared.is_dir() else project_root() / "artifacts"


def work_data_dir(environ=None, on_colab=None, home=None):
    """Returns the folder on the machine's own disk where datasets are downloaded or unpacked for a session.

    Order: the WORK_DATA_DIR environment variable; /content/data on Colab; otherwise <home>/.cache/research-data/<project>.
    It is deliberately outside the artifacts folder, so nothing in it is synced with Drive.
    """
    environ = os.environ if environ is None else environ
    if environ.get("WORK_DATA_DIR"):
        return Path(environ["WORK_DATA_DIR"])
    on_colab = os.path.isdir("/content") if on_colab is None else on_colab
    if on_colab:
        return Path("/content/data")
    return Path(home or Path.home()) / ".cache" / "research-data" / PROJECT


def check_names(exp, run_id):
    """Raises ValueError unless exp looks like exp-NNN (or scratch) and run_id like run-NNN (or smoke-NNN)."""
    if not EXP_PATTERN.match(exp):
        raise ValueError(f"experiment folder {exp!r} must be exp-NNN or scratch")
    if not RUN_PATTERN.match(run_id):
        raise ValueError(f"run id {run_id!r} must be run-NNN or smoke-NNN")
    if run_id.startswith("smoke") != (exp == "scratch"):
        raise ValueError("smoke runs live in scratch, and real runs live in an exp-NNN folder")


def exp_dir(exp, root=None):
    """Returns <root>/<exp>."""
    return Path(root or artifact_root()) / exp


def run_dir(exp, run_id, root=None):
    """Returns <root>/<exp>/<run_id> after checking the names."""
    check_names(exp, run_id)
    return exp_dir(exp, root) / run_id


def data_dir(root=None):
    """Returns <root>/data, where datasets and caches go."""
    return Path(root or artifact_root()) / "data"


def figure_dir(notebook, root=None):
    """Returns <root>/scratch/figures/<notebook>, where an exploration notebook saves its pictures."""
    return Path(root or artifact_root()) / "scratch" / "figures" / notebook
