import nox
import argparse
from pathlib import Path
import os
import tempfile
import shutil


def git_rev_parse(session, commit):
    print(f"Converting provided commit '{commit}' to Git revision...")
    rev = session.run("git", "rev-parse", commit, external=True, silent=True).strip()
    return rev


@nox.session
def save_and_load(session: nox.Session):
    """Save models and outputs to disk and compare outputs between versions.

    This session installs the bayesflow version specified by the `commit` argument, and runs the test suite either in
    "save" or in "load" mode. In save mode, results are stored to disk and a within-version load test is performed.
    In load mode, the stored models and outputs are loaded from disk, and old and new outputs are compared.
    This helps to detect breaking serialization between versions.

    Important: The test code from the current checkout, not from `commit`, is used.
    """
    # parse the arguments
    parser = argparse.ArgumentParser()
    # add subparsers for the two different commands
    subparsers = parser.add_subparsers(help="subcommand help", dest="mode")
    # save command
    parser_save = subparsers.add_parser("save")
    parser_save.add_argument("commit", type=str)
    # load command, additional "from" argument
    parser_load = subparsers.add_parser("load")
    parser_load.add_argument("--from", type=str, required=True, dest="from_commit")
    parser_load.add_argument("commit", type=str)

    # keep unknown arguments, they will be forwarded to pytest below
    args, unknownargs = parser.parse_known_args(session.posargs)

    if args.mode == "load":
        if args.from_commit == ".":
            from_commit = "local"
        else:
            from_commit = git_rev_parse(session, args.from_commit)

        from_path = Path("_compatibility_data").absolute() / from_commit
        if not from_path.exists():
            raise FileNotFoundError(
                f"The directory {from_path} does not exist, cannot load data.\n"
                f"Please run 'nox -- save {args.from_commit}' to create it, and then rerun this command."
            )

        print(f"Data will be loaded from path {from_path}.")

    # install dependencies, currently the jax backend is used, but we could add a configuration option for this
    repo_path = Path(os.curdir).absolute()
    if args.commit == ".":
        print("'.' provided, installing local state...")
        if args.mode == "save":
            print("Output will be saved to the alias 'local'")
        commit = "local"
        session.install(".[test]")
    else:
        commit = git_rev_parse(session, args.commit)
        print("Installing specified revision...")
        session.install(f"bayesflow[test] @ git+file://{str(repo_path)}@{commit}")
    session.install("jax")

    with tempfile.TemporaryDirectory() as tmpdirname:
        # launch in temporary directory, as the local bayesflow would overshadow the installed one
        tmpdirname = Path(tmpdirname)
        # pass mode and data path to pytest, required for correct save and load behavior
        if args.mode == "load":
            data_path = from_path
        else:
            data_path = Path("_compatibility_data").absolute() / commit
            if data_path.exists():
                print(f"Removing existing data directory {data_path}...")
                shutil.rmtree(data_path)

        cmd = ["pytest", "tests/test_compatibility", f"--mode={args.mode}", f"--data-path={data_path}"]
        cmd += unknownargs

        print(f"Copying tests from working directory to temporary directory: {tmpdirname}")
        shutil.copytree("tests", tmpdirname / "tests")
        with session.chdir(tmpdirname):
            session.run(*cmd, env={"KERAS_BACKEND": "jax"})
