"""Tells the code where each folder of the project lives, with one function per folder.

Code, notebooks and tools use these functions and never build paths by hand.

Three places hold one project: the vault (the notes), this code folder (synced with GitHub) and the artifacts folder
(synced with Google Drive). Artifacts are everything that is not code: data/, scratch/ and exp-NNN/run-NNN/.
On this computer that folder is C:\\Users\\<you>\\ResearchArtifacts\\<project>. On Colab it is the same folder as
seen through the mounted Drive.
"""
import os
import re
from pathlib import Path

# The name of this project. The code folder, the artifacts folder and the Drive folder all use it.
PROJECT = "face-detection-classification"

# The parent folder of the artifacts of all projects, on this computer and on Drive.
ARTIFACTS_PARENT = "ResearchArtifacts"

# Where the mounted Drive appears on a Colab machine.
DRIVE_MOUNT = Path("/content/drive")

# Experiment folders are named exp-NNN, or "scratch" for throwaway work. Run folders are named run-NNN, smoke-NNN or
# medium-NNN.
EXP_PATTERN = re.compile(r"^(exp-\d{3}|scratch)$")
RUN_PATTERN = re.compile(r"^(run|smoke|medium)-\d{3}$")


def project_root():
    """Returns the folder that holds this file.

    This is the code folder on this computer and the git clone on Colab.
    """
    return Path(__file__).resolve().parent


def drive_artifacts_dir(environ=None, mount=DRIVE_MOUNT):
    """Finds this project's artifacts folder on the mounted Drive, or returns None.

    When the DRIVE_ARTIFACTS_DIR environment variable is set, the folder it names is returned without a check.
    Otherwise the first of these folders that exists is returned. The folder of a computer synced by the Drive desktop
    app comes first, which is Othercomputers/<computer name>/ResearchArtifacts/<project>, with the computer names tried
    in alphabetical order. MyDrive/ResearchArtifacts/<project> comes last. The synced computer folder comes before
    MyDrive because that is where this computer's own copy of the artifacts lives, and the stage scripts and the status
    probe look there first too.
    """
    environ = os.environ if environ is None else environ

    # An explicit setting wins over any search.
    if environ.get("DRIVE_ARTIFACTS_DIR"):
        return Path(environ["DRIVE_ARTIFACTS_DIR"])

    # Otherwise search the synced computer folders, then My Drive, and take the first folder that exists.
    candidates = sorted((Path(mount) / "Othercomputers").glob(f"*/{ARTIFACTS_PARENT}/{PROJECT}"))
    candidates.append(Path(mount) / "MyDrive" / ARTIFACTS_PARENT / PROJECT)
    return next((c for c in candidates if c.is_dir()), None)


def artifact_root(environ=None, on_colab=None, mount=DRIVE_MOUNT, home=None):
    """Returns the artifacts folder for this machine.

    The choice follows this order. First comes the folder named by the ARTIFACT_ROOT environment variable. On Colab,
    the next choice is the Drive artifacts folder when Drive is mounted, and otherwise /content/artifacts, which is
    lost when the machine ends. On this computer, the choice is <home>/ResearchArtifacts/<project> when that folder
    exists, and otherwise <project>/artifacts. The arguments exist so that tests can set the situation.
    """
    environ = os.environ if environ is None else environ
    if environ.get("ARTIFACT_ROOT"):
        return Path(environ["ARTIFACT_ROOT"])

    # On Colab, prefer the mounted Drive so that results outlive the session.
    on_colab = os.path.isdir("/content") if on_colab is None else on_colab
    if on_colab:
        synced = drive_artifacts_dir(environ, mount)
        return synced if synced else Path("/content/artifacts")

    # On this computer, use the shared artifacts folder when it exists, and the project's own folder otherwise.
    shared = Path(home or Path.home()) / ARTIFACTS_PARENT / PROJECT
    return shared if shared.is_dir() else project_root() / "artifacts"


def work_data_dir(environ=None, on_colab=None, home=None):
    """Returns the folder on the machine's own disk where datasets are downloaded or unpacked for a session.

    The choice follows this order. First comes the folder named by the WORK_DATA_DIR environment variable. On Colab,
    the next choice is /content/data. Otherwise the choice is <home>/.cache/research-data/<project>. The folder is
    deliberately outside the artifacts folder, so nothing in it is synced with Drive.
    """
    environ = os.environ if environ is None else environ
    if environ.get("WORK_DATA_DIR"):
        return Path(environ["WORK_DATA_DIR"])
    on_colab = os.path.isdir("/content") if on_colab is None else on_colab
    if on_colab:
        return Path("/content/data")
    return Path(home or Path.home()) / ".cache" / "research-data" / PROJECT


def check_names(exp, run_id):
    """Checks the experiment folder name and the run id, and raises ValueError when either is wrong.

    The experiment folder must be named exp-NNN or scratch. The run id must be run-NNN, smoke-NNN or medium-NNN. Smoke
    and medium runs must live in scratch, and real runs must live in an exp-NNN folder.
    """
    if not EXP_PATTERN.match(exp):
        raise ValueError(f"experiment folder {exp!r} must be exp-NNN or scratch")
    if not RUN_PATTERN.match(run_id):
        raise ValueError(f"run id {run_id!r} must be run-NNN, smoke-NNN or medium-NNN")
    if run_id.startswith(("smoke", "medium")) != (exp == "scratch"):
        raise ValueError("smoke and medium runs live in scratch, and real runs live in an exp-NNN folder")


# The run-id prefix that each notebook mode uses, so that the name of a run folder says which mode made it.
RUN_PREFIX_FOR_MODE = {"smoke": "smoke", "medium": "medium", "full": "run"}


def check_run_name_for_mode(mode, run_id):
    """Raises ValueError unless run_id starts with the prefix of the notebook mode.

    The prefix is smoke- for the smoke mode, medium- for the medium mode and run- for the full mode.
    """
    prefix = RUN_PREFIX_FOR_MODE[mode]
    if not run_id.startswith(prefix + "-"):
        raise ValueError(f"MODE is {mode}, so RUN_ID must be a {prefix}-NNN name that no earlier run used, "
                         f"not {run_id!r}: change RUN_ID in the parameters cell")


def exp_dir(exp, root=None):
    """Returns the path of the experiment folder, which is <root>/<exp>."""
    return Path(root or artifact_root()) / exp


def run_dir(exp, run_id, root=None):
    """Returns the path of the run folder, which is <root>/<exp>/<run_id>, after checking both names."""
    check_names(exp, run_id)
    return exp_dir(exp, root) / run_id


def data_dir(root=None):
    """Returns the path of the data folder, which is <root>/data and is where datasets and caches go."""
    return Path(root or artifact_root()) / "data"


def figure_dir(notebook, root=None):
    """Returns <root>/scratch/figures/<notebook>, the folder where an exploration notebook saves its pictures."""
    return Path(root or artifact_root()) / "scratch" / "figures" / notebook
