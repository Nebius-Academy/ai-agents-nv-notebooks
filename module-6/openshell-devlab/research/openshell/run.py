"""Provision and run the DeepResearch agent inside NVIDIA OpenShell.

The trusted wrapper uses only Python's standard library. Research dependencies
are installed from the lockfile inside the sandbox image.

    python openshell/run.py run "Compare Linear with Jira and Asana"
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
OPENSHELL_DIR = Path(__file__).resolve().parent
POLICY_FILE = OPENSHELL_DIR / "policy.yaml"
DOCKERFILE = PROJECT_DIR / "Dockerfile.openshell"
AGENT_IMAGE = "openshell-deep-research-agent:course-1"
RUNS_DIR = PROJECT_DIR / "runs"

NEBIUS_PROFILE = OPENSHELL_DIR / "providers" / "nebius-token-factory.yaml"
TAVILY_PROFILE = OPENSHELL_DIR / "providers" / "tavily-search.yaml"
NEBIUS_PROFILE_ID = "course-nebius-token-factory"
TAVILY_PROFILE_ID = "course-tavily-search"
NEBIUS_PROVIDER = "deep-research-nebius"
TAVILY_PROVIDER = "deep-research-tavily"

SECRET_KEYS = ("TOFA_API_KEY", "TAVILY_API_KEY")
SANDBOX_CONFIG_KEYS = (
    "LEAD_MODEL",
    "WORKER_MODEL",
    "SEARCH_EXCLUDE_DOMAINS",
    "TOKEN_FACTORY_REGION",
)
REGION_PATTERN = re.compile(r"^[a-z0-9-]+$")
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,18}$")


def _display(command: list[str]) -> None:
    print(f"+ {shlex.join(command)}", flush=True)


def _run(
    command: list[str],
    *,
    env: dict[str, str],
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    _display(command)
    return subprocess.run(
        command,
        cwd=PROJECT_DIR,
        env=env,
        check=check,
        text=True,
        capture_output=capture,
    )


def _runtime_env(require_secrets: bool) -> dict[str, str]:
    """Read credentials supplied by the Devlab creation form."""
    env = os.environ.copy()
    if require_secrets:
        missing = [key for key in SECRET_KEYS if not env.get(key)]
        if missing:
            joined = ", ".join(missing)
            raise SystemExit(
                f"Missing {joined}. Supply both API keys in the Devlab creation form."
            )
    return env


def _require_openshell() -> None:
    if shutil.which("openshell") is None:
        raise SystemExit(
            "The OpenShell CLI is not installed. Install the NVIDIA binary from "
            "https://docs.nvidia.com/openshell/latest/about/installation"
        )


def _profile_exists(profile_id: str, env: dict[str, str]) -> bool:
    result = _run(
        ["openshell", "profile", "export", profile_id, "--output", "yaml"],
        env=env,
        check=False,
        capture=True,
    )
    return result.returncode == 0


def _ensure_profile(profile_id: str, profile_file: Path, env: dict[str, str]) -> None:
    # The gateway-aware linter rejects a custom profile ID that is already
    # registered, so check first to keep repeated setup/run calls idempotent.
    if _profile_exists(profile_id, env):
        return

    _run(
        ["openshell", "profile", "lint", "--file", str(profile_file)],
        env=env,
    )
    _run(
        [
            "openshell",
            "profile",
            "import",
            "--file",
            str(profile_file),
        ],
        env=env,
    )


def _provider_exists(provider_name: str, env: dict[str, str]) -> bool:
    result = _run(
        ["openshell", "provider", "get", provider_name],
        env=env,
        check=False,
        capture=True,
    )
    return result.returncode == 0


def _ensure_provider(
    provider_name: str,
    profile_id: str,
    credential_key: str,
    env: dict[str, str],
) -> None:
    # Bare credential names make OpenShell read from this process environment;
    # secret values never enter argv, terminal history, or this repository.
    if _provider_exists(provider_name, env):
        command = [
            "openshell",
            "provider",
            "update",
            provider_name,
            "--credential",
            credential_key,
        ]
    else:
        command = [
            "openshell",
            "provider",
            "create",
            "--name",
            provider_name,
            "--type",
            profile_id,
            "--credential",
            credential_key,
        ]
    _run(command, env=env)


def setup(env: dict[str, str]) -> None:
    _run(["openshell", "doctor", "check"], env=env)
    _ensure_profile(NEBIUS_PROFILE_ID, NEBIUS_PROFILE, env)
    _ensure_profile(TAVILY_PROFILE_ID, TAVILY_PROFILE, env)
    _ensure_provider(NEBIUS_PROVIDER, NEBIUS_PROFILE_ID, "TOFA_API_KEY", env)
    _ensure_provider(TAVILY_PROVIDER, TAVILY_PROFILE_ID, "TAVILY_API_KEY", env)


def build_image(env: dict[str, str]) -> None:
    """Build into the same Docker daemon used by the existing local gateway."""
    _run(
        ["docker", "build", "--tag", AGENT_IMAGE, "--file", str(DOCKERFILE), "."],
        env=env,
    )


def _render_policy(env: dict[str, str], directory: Path) -> Path:
    """Fill the Token Factory region into policy.yaml; return the rendered file."""
    region = env.get("TOKEN_FACTORY_REGION") or "us-central1"
    if not REGION_PATTERN.fullmatch(region):
        raise SystemExit(f"Invalid TOKEN_FACTORY_REGION: {region!r}")
    rendered = directory / "policy.yaml"
    rendered.write_text(
        POLICY_FILE.read_text().replace("${TOKEN_FACTORY_REGION}", region)
    )
    return rendered


def _validate_name(name: str) -> str:
    if not NAME_PATTERN.fullmatch(name):
        raise SystemExit(
            "Sandbox names must be lowercase alphanumeric/hyphen strings up to "
            "19 characters."
        )
    return name


def _generated_name() -> str:
    timestamp = datetime.now(UTC).strftime("%m%d%H%M%S")
    return f"dr-{timestamp}-{uuid.uuid4().hex[:5]}"


def _sandbox_exists(name: str, env: dict[str, str]) -> bool:
    result = _run(
        ["openshell", "sandbox", "get", name],
        env=env,
        check=False,
        capture=True,
    )
    return result.returncode == 0


def run_agent(args: argparse.Namespace, env: dict[str, str]) -> int:
    setup(env)
    build_image(env)

    sandbox_name = _validate_name(args.name or _generated_name())
    run_id = f"competitor-{uuid.uuid4().hex[:12]}"
    question = " ".join(args.question).strip()
    sandbox_env = {
        "AGENT_RUN_ID": run_id,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "UV_CACHE_DIR": "/tmp/uv-cache",
        "UV_NO_PROGRESS": "1",
    }
    sandbox_env.update((key, env[key]) for key in SANDBOX_CONFIG_KEYS if env.get(key))

    policy_dir = tempfile.TemporaryDirectory()
    policy_file = _render_policy(env, Path(policy_dir.name))

    create_command = [
        "openshell",
        "sandbox",
        "create",
        "--name",
        sandbox_name,
        "--from",
        AGENT_IMAGE,
        "--policy",
        str(policy_file),
        "--provider",
        NEBIUS_PROVIDER,
        "--provider",
        TAVILY_PROVIDER,
        "--no-auto-providers",
        "--approval-mode",
        "manual",
        "--cpu",
        args.cpu,
        "--memory",
        args.memory,
        "--label",
        "app=deep-research",
        "--label",
        "network-profile=brokered-search",
    ]
    for key, value in sandbox_env.items():
        create_command.extend(["--env", f"{key}={value}"])
    # Keep the sandbox's main process alive. OpenShell transitions a sandbox to
    # Error when that process exits (even with status 0), which would make the
    # completed run unavailable for downloads and policy demonstrations.
    create_command.extend(["--detach", "--no-tty", "--", "/usr/bin/sleep", "infinity"])

    try:
        create = _run(create_command, env=env, check=False)
    finally:
        policy_dir.cleanup()
    final_status = create.returncode

    agent_command = [
        "openshell",
        "sandbox",
        "exec",
        "--name",
        sandbox_name,
        "--no-tty",
        "--no-login-shell",
        "--",
        "uv",
        "run",
        "--frozen",
        "--no-sync",
        "python",
        "main.py",
    ]
    if question:
        agent_command.append(question)

    if create.returncode == 0:
        result = _run(agent_command, env=env, check=False)
        final_status = result.returncode

    if _sandbox_exists(sandbox_name, env):
        local_run_dir = RUNS_DIR / run_id
        local_run_dir.mkdir(parents=True, exist_ok=True)
        download = _run(
            [
                "openshell",
                "sandbox",
                "download",
                sandbox_name,
                f"/sandbox/runs/{run_id}",
                str(local_run_dir),
            ],
            env=env,
            check=False,
        )
        if final_status == 0 and download.returncode != 0:
            final_status = download.returncode

        print(f"Sandbox: {sandbox_name}")
        print(f"Local run: {local_run_dir}")

        if args.delete and download.returncode == 0:
            deletion = _run(
                ["openshell", "sandbox", "delete", sandbox_name],
                env=env,
                check=False,
            )
            if final_status == 0:
                final_status = deletion.returncode
        else:
            if args.delete:
                print("Download failed; preserving sandbox artifacts.", file=sys.stderr)
            print(f"Inspect:   python openshell/run.py inspect {sandbox_name}")
            print(f"Deny demo: python openshell/run.py demo {sandbox_name}")

    return final_status


def inspect_sandbox(name: str, env: dict[str, str]) -> int:
    name = _validate_name(name)
    policy = _run(
        ["openshell", "policy", "get", name, "--full"],
        env=env,
        check=False,
    )
    logs = _run(
        [
            "openshell",
            "logs",
            name,
            "--since",
            "30m",
            "--source",
            "sandbox",
            "-n",
            "200",
        ],
        env=env,
        check=False,
    )
    return policy.returncode or logs.returncode


def demo_denials(name: str, env: dict[str, str]) -> int:
    name = _validate_name(name)
    attempts = [
        (
            "unlisted destination (Reddit)",
            [
                "/usr/local/bin/python3.12",
                "-c",
                "import urllib.request; "
                "urllib.request.urlopen('https://www.reddit.com/', timeout=5)",
            ],
            ("NET:OPEN", "DENIED", "/usr/local/bin/python3.12(",
             " -> www.reddit.com:443 ", "reason:transparent_tcp_policy_denied"),
        ),
        (
            "approved destination from an unapproved binary (curl -> Tavily)",
            [
                "curl",
                "-fsS",
                "--max-time",
                "5",
                "-X",
                "POST",
                "https://api.tavily.com/search",
            ],
            ("NET:OPEN", "DENIED", "/usr/bin/curl(",
             " -> api.tavily.com:443 ", "reason:transparent_tcp_policy_denied"),
        ),
        (
            "approved binary and host with an unapproved method (Python GET)",
            [
                "uv",
                "run",
                "--frozen",
                "--no-sync",
                "python",
                "-c",
                (
                    "import urllib.request; "
                    "urllib.request.urlopen('https://api.tavily.com/search', "
                    "timeout=5)"
                ),
            ],
            ("HTTP:GET", "DENIED", "L7_REQUEST deny GET api.tavily.com:443/search"),
        ),
    ]

    logs_command = [
        "openshell", "logs", name, "--since", "5m", "--source", "sandbox", "-n", "500",
        "--color", "never",
    ]
    failed = False
    for label, attempted_command, expected_fields in attempts:
        before = _run(logs_command, env=env, check=False, capture=True)
        if before.returncode != 0:
            print("Cannot read policy audit logs; denial demo is inconclusive.", file=sys.stderr)
            return 1
        previous_lines = set(before.stdout.splitlines())
        print(f"\nExpected denial: {label}")
        result = _run(
            [
                "openshell",
                "sandbox",
                "exec",
                "--name",
                name,
                "--no-tty",
                "--no-login-shell",
                "--",
                *attempted_command,
            ],
            env=env,
            check=False,
        )
        if result.returncode == 0:
            failed = True
            print("ERROR: request unexpectedly succeeded", file=sys.stderr)
            continue
        # Nonzero alone could mean DNS, TLS, or execution failure. Require a
        # fresh matching OpenShell v0.1.1 policy event, not an older denial.
        for attempt in range(5):
            logs = _run(logs_command, env=env, check=False, capture=True)
            fresh_lines = set(logs.stdout.splitlines()) - previous_lines
            evidence = [
                line for line in fresh_lines if all(field in line for field in expected_fields)
            ]
            if logs.returncode == 0 and evidence:
                print("\n".join(evidence))
                break
            if attempt < 4:
                time.sleep(1)
        else:
            failed = True
            print("No matching new policy denial; demo is inconclusive.", file=sys.stderr)
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("setup", help="install profiles and refresh providers")
    subparsers.add_parser("build", help="build the course image in the local Docker daemon")

    run_parser = subparsers.add_parser("run", help="run the agent in OpenShell")
    run_parser.add_argument("--name", help="explicit sandbox name")
    run_parser.add_argument("--cpu", default="2", help="sandbox CPU limit")
    run_parser.add_argument("--memory", default="4Gi", help="sandbox memory limit")
    run_parser.add_argument(
        "--delete",
        action="store_true",
        help="delete the sandbox after downloading run artifacts",
    )
    run_parser.add_argument("question", nargs="*", help="research question")

    inspect_parser = subparsers.add_parser(
        "inspect", help="print effective policy and recent audit logs"
    )
    inspect_parser.add_argument("name")

    demo_parser = subparsers.add_parser(
        "demo", help="exercise host, binary, and HTTP-method denials"
    )
    demo_parser.add_argument("name")

    delete_parser = subparsers.add_parser("delete", help="delete a sandbox")
    delete_parser.add_argument("name")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    _require_openshell()
    require_secrets = args.command in {"setup", "run"}
    env = _runtime_env(require_secrets=require_secrets)

    if args.command == "setup":
        setup(env)
        return 0
    if args.command == "build":
        build_image(env)
        return 0
    if args.command == "run":
        return run_agent(args, env)
    if args.command == "inspect":
        return inspect_sandbox(args.name, env)
    if args.command == "demo":
        return demo_denials(args.name, env)
    if args.command == "delete":
        name = _validate_name(args.name)
        return _run(
            ["openshell", "sandbox", "delete", name], env=env, check=False
        ).returncode
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
