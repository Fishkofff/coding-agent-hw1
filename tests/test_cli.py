"""Сквозные тесты CLI: отдельные процессы, настоящий сбой процесса (os._exit) и продолжение по run_id."""
import os
import re
import subprocess
import sys
import unittest

from tests.helpers import FIXED_ORDER, ORIGINAL_ORDER, ROOT, Project


class CliTest(unittest.TestCase):
    def setUp(self):
        self.p = Project(self)

    def cli(self, *args, expect=0):
        proc = subprocess.run([sys.executable, "-m", "agent", *args, "--policy", str(self.p.policy_path)],
                              cwd=ROOT, capture_output=True, text=True, timeout=120,
                              env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(self.p.root)})
        self.assertEqual(proc.returncode, expect, proc.stdout + proc.stderr)
        return proc.stdout + proc.stderr

    @staticmethod
    def run_id(output):
        return re.search(r"run_id: (\S+)", output).group(1)

    def test_scripted_run_diff_and_rollback(self):
        out = self.cli("run", "Исправь расчёт", "--mode", "auto", "--no-plan",
                       "--scripted", str(ROOT / "demo" / "scripted_no_plan.json"))
        run_id = self.run_id(out)
        self.assertIn("completed", out)
        self.assertEqual(self.p.read("order.py"), FIXED_ORDER)
        diff = self.cli("diff", run_id)
        self.assertIn("MODIFIED order.py", diff)
        self.assertIn("+    if not 0 <= discount_percent <= 100:", diff)
        self.cli("rollback", run_id)
        self.assertEqual(self.p.read("order.py"), ORIGINAL_ORDER)

    def test_plan_mode_with_confirmations_across_processes(self):
        scripted = str(ROOT / "demo" / "scripted_plan.json")
        out = self.cli("run", "Исправь расчёт", "--scripted", scripted)          # safe + plan mode по умолчанию
        run_id = self.run_id(out)
        self.assertIn("waiting (plan_approval)", out)
        self.assertIn("План v1", self.cli("plan", "show", run_id))
        self.assertIn("waiting (confirmation)", self.cli("plan", "approve", run_id))
        self.assertIn("waiting (confirmation)", self.cli("approve", run_id))
        out = self.cli("approve", run_id)
        self.assertIn("- [x] T3: Запустить тесты", out)
        self.assertIn("completed", out)
        tasks = self.cli("tasks", run_id)
        self.assertIn("подтверждение runtime", tasks)

    def test_real_process_crash_and_recovery(self):
        scripted = str(ROOT / "demo" / "scripted_no_plan.json")
        out = self.cli("run", "fix", "--mode", "auto", "--no-plan", "--scripted", scripted,
                       "--crash-at", "after_tool_execute:write_file", expect=137)
        run_id = self.run_id(out)
        self.assertIn("процесс прервался", self.cli("status", run_id))
        out = self.cli("resume", run_id)
        self.assertIn("waiting (recovery)", out)
        self.assertIn("M order.py", out)
        self.assertIn("--changes", self.cli("recover", run_id, "--action", "retry", expect=2))
        out = self.cli("recover", run_id, "--action", "retry", "--changes", "attribute")
        self.assertIn("completed", out)
        self.assertIn("attempts=2", self.cli("calls", run_id))

    def test_policy_check_command(self):
        self.assertIn("ОТКЛОНЕНО", self.cli("policy-check", "Bash", '{"command": "ls | cat", "cwd": "."}', expect=1))
        self.assertIn("ОТКЛОНЕНО", self.cli("policy-check", "read_file", '{"path": "../policy.json"}', expect=1))
        out = self.cli("policy-check", "delete_file", '{"path": "order.py"}', "--mode", "auto")
        self.assertIn("нужно подтверждение", out)


if __name__ == "__main__":
    unittest.main()
