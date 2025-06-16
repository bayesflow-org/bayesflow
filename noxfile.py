import nox
import argparse
from pathlib import Path
import os


@nox.session
def save_and_load(session: nox.Session):
    """Save models and outputs to disk and compare outputs between versions.

    This session installs the bayesflow version specified by the `commit` argument, and runs the test suite either in
    "save" or in "load" mode. In save mode, results are stored to disk and a within-version load test is performed.
    In load mode, the stored models and outputs are loaded from disk, and old and new outputs are compared.
    This helps to detect breaking serialization between versions.

    Important: The test code from the current checkout is used, not from the installed version.
    """
    # parse the arguments
    parser = argparse.ArgumentParser()
    # add subparsers for the two different commands
    subparsers = parser.add_subparsers(help="subcommand help", dest="mode")
    # save command
    parser_save = subparsers.add_parser("save")
    parser_save.add_argument("commit", type=str, default=".")
    # load command, additional "from" argument
    parser_load = subparsers.add_parser("load")
    parser_load.add_argument("commit", type=str, default=".")
    parser.add_argument("--from", type=str, default="", required=False, dest="from_commit")
    # keep unknown arguments, they will be forwarded to pytest below
    args, unknownargs = parser.parse_known_args(session.posargs)

    # install dependencies, currently the jax backend is used, but we could add a configuration option for this
    repo_path = Path(os.curdir).absolute().parent / "bf2"
    session.install(f"git+file://{str(repo_path)}@{args.commit}")
    session.install("jax")
    session.install("pytest")

    # pass mode and commits to pytest, required for correct save and load behavior
    cmd = ["pytest", "--mode", args.mode, "--commit", args.commit]
    if args.mode == "load":
        cmd += ["--from", args.from_commit]
    cmd += unknownargs

    session.run(*cmd, env={"KERAS_BACKEND": "jax"})
