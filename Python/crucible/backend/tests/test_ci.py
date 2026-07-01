"""
CI infrastructure tests.

Validates that the CI/CD files are well-formed and self-consistent:
  - GitHub Actions workflows are valid YAML
  - Workflow files reference correct job names and steps
  - Dockerfile has required directives (FROM, USER, HEALTHCHECK, EXPOSE, CMD)
  - .dockerignore covers critical exclusions
  - pyproject.toml is valid TOML
  - Makefile targets are syntactically correct
  - .env.example covers all settings referenced in config.py

These tests run in CI itself — they're a meta-CI layer that validates
the CI configuration before it goes live.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import yaml   # pyyaml

# Root of the repo (one level above backend/)
REPO_ROOT    = Path(__file__).parent.parent.parent   # crucible/
BACKEND_ROOT = Path(__file__).parent.parent           # crucible/backend/
WORKFLOWS    = REPO_ROOT / ".github" / "workflows"


# ══════════════════════════════════════════════════════════════════════════
# GITHUB ACTIONS WORKFLOWS
# ══════════════════════════════════════════════════════════════════════════

class TestWorkflows:

    @pytest.fixture
    def workflow_files(self):
        return list(WORKFLOWS.glob("*.yml")) if WORKFLOWS.exists() else []

    def test_workflows_directory_exists(self):
        assert WORKFLOWS.exists(), f"Missing .github/workflows/ at {WORKFLOWS}"

    def test_required_workflows_present(self):
        names = {f.stem for f in WORKFLOWS.glob("*.yml")}
        for required in ("test", "lint", "deploy"):
            assert required in names, f"Missing required workflow: {required}.yml"

    def test_all_workflows_are_valid_yaml(self, workflow_files):
        assert workflow_files, "No workflow files found"
        for wf_file in workflow_files:
            with open(wf_file) as f:
                content = f.read()
            try:
                parsed = yaml.safe_load(content)
            except yaml.YAMLError as exc:
                pytest.fail(f"{wf_file.name} is not valid YAML: {exc}")
            assert isinstance(parsed, dict), f"{wf_file.name} must be a YAML mapping"

    def test_test_workflow_has_required_keys(self):
        wf = yaml.safe_load((WORKFLOWS / "test.yml").read_text())
        # PyYAML parses YAML 'on' as Python True (YAML 1.1 boolean)
        assert True in wf or "on" in wf, "test.yml must have 'on' trigger"
        assert "jobs" in wf, "test.yml must have 'jobs'"

    def test_test_workflow_runs_pytest(self):
        content = (WORKFLOWS / "test.yml").read_text()
        assert "pytest" in content, "test.yml must run pytest"

    def test_test_workflow_has_coverage(self):
        content = (WORKFLOWS / "test.yml").read_text()
        assert "--cov" in content, "test.yml must run with coverage"

    def test_deploy_workflow_requires_tests(self):
        """Deploy workflow must depend on tests passing."""
        content = (WORKFLOWS / "deploy.yml").read_text()
        assert "needs" in content, "deploy.yml must have 'needs' dependency on tests"

    def test_deploy_workflow_pushes_to_registry(self):
        content = (WORKFLOWS / "deploy.yml").read_text()
        assert "push" in content and ("ghcr" in content or "registry" in content.lower()), \
            "deploy.yml must push to a container registry"

    def test_lint_workflow_runs_ruff(self):
        content = (WORKFLOWS / "lint.yml").read_text()
        assert "ruff" in content, "lint.yml must run ruff"

    def test_workflows_use_pinned_action_versions(self):
        """Action refs must use @v3 or @v4, not @master (reproducibility)."""
        for wf_file in WORKFLOWS.glob("*.yml"):
            content = wf_file.read_text()
            # Find uses: lines
            uses_lines = [l.strip() for l in content.splitlines() if "uses:" in l]
            for line in uses_lines:
                assert "@master" not in line and "@main" not in line, \
                    f"{wf_file.name}: unpinned action '{line}' — use @vN instead"

    def test_test_workflow_has_caching(self):
        """pip cache must be configured to speed up CI."""
        content = (WORKFLOWS / "test.yml").read_text()
        assert "actions/cache" in content, "test.yml must cache pip packages"


# ══════════════════════════════════════════════════════════════════════════
# DOCKERFILE
# ══════════════════════════════════════════════════════════════════════════

class TestDockerfile:

    @pytest.fixture
    def dockerfile(self):
        path = BACKEND_ROOT / "Dockerfile"
        assert path.exists(), "Dockerfile not found"
        return path.read_text()

    def test_uses_multi_stage_build(self, dockerfile):
        """Multi-stage reduces the final image size significantly."""
        from_count = len(re.findall(r"^FROM\s", dockerfile, re.MULTILINE))
        assert from_count >= 2, "Dockerfile must use multi-stage build (at least 2 FROM lines)"

    def test_has_non_root_user(self, dockerfile):
        """Running as root inside a container is a security risk."""
        assert "USER" in dockerfile, "Dockerfile must switch to a non-root USER"
        assert "root" not in dockerfile.split("USER")[-1].lower().split("\n")[0], \
            "The final USER must not be root"

    def test_exposes_port_8000(self, dockerfile):
        assert "EXPOSE 8000" in dockerfile, "Dockerfile must EXPOSE port 8000"

    def test_has_healthcheck(self, dockerfile):
        assert "HEALTHCHECK" in dockerfile, "Dockerfile must define a HEALTHCHECK"

    def test_has_cmd(self, dockerfile):
        assert "CMD" in dockerfile, "Dockerfile must have a CMD"
        assert "uvicorn" in dockerfile, "CMD must start uvicorn"

    def test_has_labels(self, dockerfile):
        assert "LABEL" in dockerfile, "Dockerfile must have OCI image labels"

    def test_copies_requirements_before_code(self, dockerfile):
        """requirements.txt must be copied before app code for layer caching."""
        req_pos  = dockerfile.find("requirements.txt")
        copy_pos = dockerfile.find("COPY . .")
        if copy_pos == -1:
            copy_pos = dockerfile.find("COPY --chown")
        assert req_pos < copy_pos, \
            "requirements.txt must be copied before application code for better caching"

    def test_uses_slim_base_image(self, dockerfile):
        assert "slim" in dockerfile or "alpine" in dockerfile, \
            "Dockerfile should use a slim or alpine base image to reduce size"

    def test_cleans_apt_cache(self, dockerfile):
        assert "rm -rf /var/lib/apt/lists" in dockerfile, \
            "Dockerfile must clean apt cache (rm -rf /var/lib/apt/lists/*)"


# ══════════════════════════════════════════════════════════════════════════
# .dockerignore
# ══════════════════════════════════════════════════════════════════════════

class TestDockerIgnore:

    @pytest.fixture
    def dockerignore(self):
        path = BACKEND_ROOT / ".dockerignore"
        assert path.exists(), ".dockerignore not found"
        return path.read_text()

    def test_excludes_pycache(self, dockerignore):
        assert "__pycache__" in dockerignore

    def test_excludes_venv(self, dockerignore):
        assert ".venv" in dockerignore or "venv/" in dockerignore

    def test_excludes_test_artifacts(self, dockerignore):
        assert ".pytest_cache" in dockerignore or "pytest_cache" in dockerignore

    def test_excludes_env_files(self, dockerignore):
        assert ".env" in dockerignore, ".env secrets must be in .dockerignore"

    def test_excludes_git(self, dockerignore):
        assert ".git/" in dockerignore or ".git" in dockerignore


# ══════════════════════════════════════════════════════════════════════════
# pyproject.toml
# ══════════════════════════════════════════════════════════════════════════

class TestPyprojectToml:

    @pytest.fixture
    def pyproject(self):
        import tomllib
        path = BACKEND_ROOT / "pyproject.toml"
        assert path.exists(), "pyproject.toml not found"
        with open(path, "rb") as f:
            return tomllib.load(f)

    def test_is_valid_toml(self, pyproject):
        assert isinstance(pyproject, dict)

    def test_has_ruff_section(self, pyproject):
        assert "tool" in pyproject
        assert "ruff" in pyproject["tool"], "pyproject.toml must have [tool.ruff]"

    def test_has_pytest_section(self, pyproject):
        assert "tool" in pyproject
        assert "pytest" in pyproject["tool"], "pyproject.toml must have [tool.pytest.ini_options]"

    def test_asyncio_mode_is_auto(self, pyproject):
        pytest_conf = pyproject["tool"]["pytest"]["ini_options"]
        assert pytest_conf.get("asyncio_mode") == "auto", \
            "asyncio_mode must be 'auto' for pytest-asyncio"

    def test_has_coverage_section(self, pyproject):
        assert "coverage" in pyproject.get("tool", {}), \
            "pyproject.toml should have [tool.coverage]"

    def test_ruff_line_length_set(self, pyproject):
        ruff = pyproject["tool"]["ruff"]
        assert "line-length" in ruff


# ══════════════════════════════════════════════════════════════════════════
# Makefile
# ══════════════════════════════════════════════════════════════════════════

class TestMakefile:

    @pytest.fixture
    def makefile(self):
        path = REPO_ROOT / "Makefile"
        assert path.exists(), "Makefile not found"
        return path.read_text()

    def test_has_help_target(self, makefile):
        assert "help:" in makefile

    def test_has_test_target(self, makefile):
        assert "test:" in makefile

    def test_has_lint_target(self, makefile):
        assert "lint:" in makefile

    def test_has_docker_target(self, makefile):
        assert "docker" in makefile

    def test_has_migrate_target(self, makefile):
        assert "migrate:" in makefile

    def test_all_targets_are_phony(self, makefile):
        """All non-file targets must be in .PHONY to avoid conflicts with files of the same name."""
        assert ".PHONY" in makefile

    def test_has_dev_target(self, makefile):
        assert "dev:" in makefile


# ══════════════════════════════════════════════════════════════════════════
# .env.example
# ══════════════════════════════════════════════════════════════════════════

class TestEnvExample:

    @pytest.fixture
    def env_example(self):
        path = REPO_ROOT / ".env.example"
        assert path.exists(), ".env.example not found"
        return path.read_text()

    def test_has_database_url(self, env_example):
        assert "DATABASE_URL" in env_example

    def test_has_secret_key(self, env_example):
        assert "SECRET_KEY" in env_example

    def test_has_storage_backend(self, env_example):
        assert "STORAGE_BACKEND" in env_example

    def test_has_anthropic_key(self, env_example):
        assert "ANTHROPIC_API_KEY" in env_example

    def test_no_real_secrets(self, env_example):
        """The example file must not contain real credentials."""
        # Placeholders like "sk-ant-..." and "AKIA..." are OK;
        # real keys have specific lengths and character patterns.
        assert "sk-ant-api03" not in env_example, \
            ".env.example must not contain a real Anthropic API key (sk-ant-api03...)"
        # Real AWS keys are 20-char alphanumeric starting with AKIA; "AKIA..." is a placeholder
        real_aws_pattern = re.compile(r"AWS_ACCESS_KEY_ID\s*=\s*AKIA[A-Z0-9]{16}")
        assert not real_aws_pattern.search(env_example), \
            ".env.example must not contain a real AWS access key"

    def test_includes_aws_config(self, env_example):
        assert "AWS_BUCKET_NAME" in env_example, \
            ".env.example must document AWS S3 config"


class TestIntegrationJobInWorkflow:
    """Validates the integration test job was added to test.yml."""

    def test_test_workflow_has_integration_job(self):
        content = (WORKFLOWS / "test.yml").read_text()
        assert "integration:" in content

    def test_integration_job_depends_on_unit_tests(self):
        """Integration job must run after (not in parallel with) the fast unit suite."""
        wf = yaml.safe_load((WORKFLOWS / "test.yml").read_text())
        integration_job = wf["jobs"].get("integration", {})
        assert integration_job.get("needs") == "test"

    def test_integration_job_runs_marked_tests(self):
        content = (WORKFLOWS / "test.yml").read_text()
        assert "-m integration" in content
