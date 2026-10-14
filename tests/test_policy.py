"""Блок 2: policy, режимы safe/auto, подтверждения."""
import json
import unittest

from agent.policy import PolicyEngine, PolicyError, PolicyViolation, approval_requirement, load_policy
from agent.tools import ToolExecutor
from tests.helpers import (FIXED_ORDER, ORIGINAL_ORDER, TEST_CMD, Project, act, call, say, tool_results)


class PolicyEngineTest(unittest.TestCase):
    def setUp(self):
        self.p = Project(self, allow=[TEST_CMD, "ls", "rm order.py"], deny=["rm", "bash -c"],
                         dirs_deny=["workspace/private"],
                         extra_files={"private/secret.txt": "token", "sub/notes.txt": "notes"})
        self.engine = PolicyEngine(self.p.policy_path)
        self.policy = self.engine.load()

    def check(self, command, cwd="."):
        return self.engine.check_command(self.policy, command, cwd)

    def test_allowed_command(self):
        tokens, cwd = self.check(TEST_CMD)
        self.assertEqual(tokens, ("python3", "-m", "unittest", "-v"))
        self.assertEqual(cwd, self.p.ws)
        self.check("ls", "sub")

    def test_denied_commands(self):
        cases = [
            "rm -rf /",                                   # deny
            "python3 -m unittest -v | cat",               # конвейер
            "python3 -m unittest -v; rm order.py",        # цепочка
            "python3 -m unittest -v && ls",
            "ls $(pwd)", "ls `pwd`",                      # подстановки
            "ls > out.txt", "ls < order.py",              # перенаправления
            "bash -c ls", "bash -c 'rm -rf /'",           # вложенный shell
            "FOO=1 ls",                                   # переменные окружения
            "ls\nrm order.py",                            # перевод строки
            "ls *", "ls ~",                               # глоб, домашний каталог
            "cat order.py",                               # не в allowlist
            "python3 -m unittest",                        # не точное совпадение с разрешённой строкой
        ]
        for command in cases:
            with self.subTest(command=command), self.assertRaises(PolicyViolation):
                self.check(command)

    def test_deny_has_priority_over_allow(self):
        self.assertIn(("rm", "order.py"), self.policy.allow_commands)
        with self.assertRaisesRegex(PolicyViolation, "deny rule 'rm'"):
            self.check("rm order.py")

    def test_allowed_command_in_denied_directory(self):
        with self.assertRaisesRegex(PolicyViolation, "denied directory"):
            self.check("ls", "private")
        with self.assertRaisesRegex(PolicyViolation, "denied directory"):
            self.check(TEST_CMD, "private")

    def test_denied_subdirectory_for_file_tools(self):
        cases = [("read_file", {"path": "private/secret.txt"}),
                 ("write_file", {"path": "private/new.txt", "content": "x"}),
                 ("delete_file", {"path": "private/secret.txt"}),
                 ("list_files", {"path": "private"})]
        for tool, args in cases:
            with self.subTest(tool=tool), self.assertRaisesRegex(PolicyViolation, "denied directory"):
                self.engine.check_call(self.policy, tool, args)
        self.engine.check_call(self.policy, "read_file", {"path": "sub/notes.txt"})   # соседний каталог разрешён

    def test_list_files_hides_denied_directory_content(self):
        checked = self.engine.check_call(self.policy, "list_files", {"path": "."})
        entries = ToolExecutor().execute(checked, self.policy, 10_000)["entries"]
        self.assertIn("sub/notes.txt", entries)
        self.assertNotIn("secret", entries)

    def test_dotdot_absolute_paths_and_symlinks_are_rejected(self):
        (self.p.root / "outside.txt").write_text("outside secret")
        (self.p.ws / "link_out").symlink_to(self.p.root / "outside.txt")
        (self.p.ws / "link_in").symlink_to(self.p.ws / "order.py")
        (self.p.ws / "dir_link").symlink_to(self.p.root, target_is_directory=True)
        cases = {
            "../policy.json": "'..'",
            "sub/../../policy.json": "'..'",
            "sub/../order.py": "'..'",
            str(self.p.ws / "order.py"): "absolute",
            "/etc/passwd": "absolute",
            "~/.ssh/id_rsa": "absolute",
            "link_out": "symlink",
            "link_in": "symlink",
            "dir_link/policy.json": "symlink",
        }
        for path, reason in cases.items():
            for tool in ("read_file", "write_file"):
                args = {"path": path, "content": "x"} if tool == "write_file" else {"path": path}
                with self.subTest(path=path, tool=tool), self.assertRaisesRegex(PolicyViolation, reason):
                    self.engine.check_call(self.policy, tool, args)
        with self.assertRaisesRegex(PolicyViolation, "'..'"):
            self.check("ls", "..")
        with self.assertRaisesRegex(PolicyViolation, "symlink"):
            self.check("ls", "dir_link")

    def test_policy_file_must_be_outside_workspace(self):
        inside = self.p.ws / "policy.json"
        inside.write_text(json.dumps({**self.p.policy, "workspace": ".", "runs_dir": "../runs2"}))
        with self.assertRaisesRegex(PolicyError, "outside the workspace"):
            load_policy(inside)

    def test_policy_rejects_bad_configuration(self):
        def broken(**changes):
            data = json.loads(json.dumps(self.p.policy))
            for key, value in changes.items():
                section, _, field = key.partition("__")
                if field:
                    data[section][field] = value
                else:
                    data[section] = value
            self.p.policy_path.write_text(json.dumps(data))
            return self.p.policy_path

        with self.assertRaisesRegex(PolicyError, "outside the workspace"):
            load_policy(broken(directories__allow=["."]))
        with self.assertRaisesRegex(PolicyError, "shell syntax"):
            load_policy(broken(commands__allow=["ls | cat"]))
        with self.assertRaisesRegex(PolicyError, "hide"):
            load_policy(broken(journal={"exclude": ["*.py"]}))
        with self.assertRaisesRegex(PolicyError, "runs_dir"):
            load_policy(broken(runs_dir="workspace/runs"))

    def test_mode_matrix(self):
        expected = {
            ("read", "safe", "execution"): "auto", ("write", "safe", "execution"): "confirm",
            ("exec", "safe", "execution"): "confirm", ("delete", "safe", "execution"): "confirm",
            ("read", "auto", "execution"): "auto", ("write", "auto", "execution"): "auto",
            ("exec", "auto", "execution"): "auto", ("delete", "auto", "execution"): "confirm",
            ("read", "auto", "planning"): "auto", ("write", "auto", "planning"): "forbidden",
            ("exec", "safe", "planning"): "forbidden", ("delete", "auto", "planning"): "forbidden",
        }
        for (effect, mode, phase), requirement in expected.items():
            with self.subTest(effect=effect, mode=mode, phase=phase):
                self.assertEqual(approval_requirement(effect, mode, phase), requirement)


class ModesAndConfirmationsTest(unittest.TestCase):
    def test_safe_mode_stops_for_confirmation_and_same_run_continues(self):
        p = Project(self)
        steps = [act(call("read_file", "r1", path="order.py"),
                     call("write_file", "w1", path="order.py", content=FIXED_ORDER)),
                 act(call("Bash", "b1", command=TEST_CMD, cwd=".")),
                 say("done")]
        run_id = p.runtime(steps).run("fix", mode="safe", plan_mode=False)
        st = p.state(run_id)
        self.assertEqual((st["status"], st["pending"]["kind"], st["pending"]["call_id"]), ("waiting", "confirmation", "w1"))
        self.assertEqual(st["execution_calls"]["r1"]["status"], "completed")    # чтение — сразу
        self.assertEqual(p.read("order.py"), ORIGINAL_ORDER)                     # запись — ждёт

        p.runtime(steps).approve(run_id)                  # новый экземпляр runtime = новый процесс
        st = p.state(run_id)
        self.assertEqual(p.read("order.py"), FIXED_ORDER)
        self.assertEqual(st["pending"]["call_id"], "b1")  # Bash в safe тоже ждёт подтверждения

        p.runtime(steps).approve(run_id)
        st = p.state(run_id)
        self.assertEqual(st["status"], "completed")
        self.assertEqual(st["execution_calls"]["w1"]["decision"]["approved"], True)
        self.assertEqual(tool_results(st)["b1"]["exit_code"], 0)

    def test_denial_reason_reaches_the_model_and_it_changes_course(self):
        p = Project(self)
        seen = {}

        def after_denial(messages, tools):
            seen["result"] = json.loads(messages[-1]["content"])
            return act(call("read_file", "r2", path="test_order.py"))

        steps = [act(call("write_file", "w1", path="order.py", content="broken")), after_denial, say("ok")]
        run_id = p.runtime(steps).run("t", mode="safe", plan_mode=False)
        p.runtime(steps).deny(run_id, "не трогай order.py")
        st = p.state(run_id)
        self.assertEqual(st["status"], "completed")
        self.assertEqual(st["execution_calls"]["w1"]["status"], "denied")
        self.assertEqual(seen["result"]["error"], "denied_by_user")
        self.assertEqual(seen["result"]["reason"], "не трогай order.py")
        self.assertEqual(p.read("order.py"), ORIGINAL_ORDER)

    def test_auto_mode_runs_write_and_bash_immediately_but_asks_before_delete(self):
        p = Project(self, extra_files={"tmp.txt": "x"})
        steps = [act(call("write_file", "w1", path="order.py", content=FIXED_ORDER),
                     call("Bash", "b1", command=TEST_CMD, cwd="."),
                     call("delete_file", "d1", path="tmp.txt")),
                 say("ok")]
        run_id = p.runtime(steps).run("t", mode="auto", plan_mode=False)
        st = p.state(run_id)
        self.assertEqual(st["execution_calls"]["w1"]["status"], "completed")
        self.assertEqual(st["execution_calls"]["b1"]["status"], "completed")
        self.assertEqual(st["pending"]["call_id"], "d1")
        self.assertTrue((p.ws / "tmp.txt").exists())
        p.runtime(steps).approve(run_id)
        self.assertFalse((p.ws / "tmp.txt").exists())
        self.assertEqual(p.state(run_id)["status"], "completed")

    def test_runtime_rejects_forbidden_calls_from_the_model(self):
        p = Project(self, extra_files={"private/secret.txt": "token"})
        ex = ToolExecutor()
        steps = [act(call("Bash", "b1", command="rm order.py", cwd="."),
                     call("Bash", "b2", command="ls", cwd="private"),
                     call("Bash", "b3", command="python3 -m unittest -v && rm order.py", cwd="."),
                     call("read_file", "r1", path="../policy.json"),
                     call("write_file", "w1", path="private/x.txt", content="x")),
                 say("ok")]
        run_id = p.runtime(steps, executor=ex).run("t", mode="auto", plan_mode=False)
        results = tool_results(p.state(run_id))
        for call_id in ("b1", "b2", "b3", "r1", "w1"):
            self.assertEqual(results[call_id]["error"], "policy_denied", call_id)
        self.assertEqual(ex.executions, [])
        self.assertEqual(p.read("order.py"), ORIGINAL_ORDER)

    def test_policy_is_rechecked_right_before_execution(self):
        p = Project(self)
        steps = [act(call("Bash", "b1", command=TEST_CMD, cwd=".")), say("ok")]
        run_id = p.runtime(steps).run("t", mode="safe", plan_mode=False)
        p.policy["commands"]["deny"].append("unittest")      # policy поменяли, пока ждали подтверждения
        p.write_policy()
        ex = ToolExecutor()
        p.runtime(steps, executor=ex).approve(run_id)
        st = p.state(run_id)
        self.assertEqual(tool_results(st)["b1"]["error"], "policy_denied")
        self.assertEqual(ex.executions, [])


if __name__ == "__main__":
    unittest.main()
