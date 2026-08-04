import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_windows_runner_is_one_shot_detached_and_placeholder_free():
    path = ROOT / "scripts" / "run_p1_analysis_campaign.ps1"
    text = path.read_text(encoding="utf-8")
    assert "$MyInvocation.InvocationName -eq '.'" in text
    assert "Set-StrictMode" in text
    assert "symbolic-ref -q HEAD" in text
    assert "--expected-commit" in text
    assert "--expected-tree" in text
    assert "RecoveryStatePath" in text
    assert "--recovery-state $recoveryState" in text
    assert "--remote-branch origin/codex/vea-p1-t004a-local-r2" in text
    assert "--resume-or-start" in text
    assert "--run-all" in text
    assert "PLACEHOLDER" not in text
    assert "pip install" not in text
    assert "scripts\\run_p1_analysis_campaign.sh" not in text


def test_windows_runner_parses_with_native_powershell():
    path = ROOT / "scripts" / "run_p1_analysis_campaign.ps1"
    command = (
        "$errors=$null; $tokens=$null; "
        f"[System.Management.Automation.Language.Parser]::ParseFile('{path}',[ref]$tokens,[ref]$errors) | Out-Null; "
        "if($errors.Count -ne 0){$errors | ForEach-Object {Write-Error $_}; exit 1}"
    )
    result = subprocess.run(
        ["powershell", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
