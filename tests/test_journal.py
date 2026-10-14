"""Уровень 10: журнал изменений по run_id и безопасный откат."""
import unittest

from agent.faults import SimulatedCrash
from agent.journal import RollbackConflict, RollbackError
from agent.runtime import UserError
from agent.tools import ToolExecutor
from tests.helpers import (FIXED_ORDER, ORIGINAL_ORDER, ORIGINAL_TEST, TEST_CMD, Project, act, call, say,
                           sha_bytes)

FIX_STEPS = [act(call("write_file", "w1", path="order.py", content=FIXED_ORDER)),
             act(call("Bash", "b1", command=TEST_CMD, cwd=".")),
             say("done")]


class JournalAndRollbackTest(unittest.TestCase):
    def test_diff_and_rollback_after_restart(self):
        p = Project(self)
        run_id = p.runtime(FIX_STEPS).run("fix", mode="auto", plan_mode=False)

        report = p.runtime([]).diff_report(run_id)            # «после перезапуска»: новый экземпляр
        self.assertEqual([f["path"] for f in report["files"]], ["order.py"])
        change = report["files"][0]
        self.assertEqual(change["op"], "modified")
        self.assertEqual(change["before"], sha_bytes(ORIGINAL_ORDER.encode()))
        self.assertEqual(change["after"], sha_bytes(FIXED_ORDER.encode()))
        self.assertFalse(change["externally_modified"])
        self.assertIn("-    return sum(prices) - discount_percent", change["diff"])
        self.assertIn("+    if not 0 <= discount_percent <= 100:", change["diff"])
        kinds = [(e["kind"], e["tool"], [c["path"] for c in e["changes"]]) for e in report["entries"]]
        self.assertEqual(kinds, [("action", "write_file", ["order.py"]), ("action", "Bash", [])])

        (p.ws / "notes.txt").write_text("my own file")         # посторонний файл, не относится к запуску
        result = p.runtime([]).rollback(run_id)
        self.assertEqual(result["restored"], ["order.py"])
        self.assertEqual(p.read("order.py"), ORIGINAL_ORDER)
        self.assertEqual(p.read("test_order.py"), ORIGINAL_TEST)
        self.assertEqual(p.read("notes.txt"), "my own file")
        self.assertEqual(p.state(run_id)["status"], "rolled_back")

    def test_external_change_makes_rollback_a_conflict_and_nothing_is_written(self):
        p = Project(self)
        steps = [act(call("write_file", "w1", path="order.py", content=FIXED_ORDER),
                     call("write_file", "w2", path="helper.py", content="HELPER = 1\n")),
                 say("done")]
        run_id = p.runtime(steps).run("fix", mode="auto", plan_mode=False)
        external = FIXED_ORDER + "# edited by a human\n"
        (p.ws / "order.py").write_text(external)

        report = p.runtime([]).diff_report(run_id)
        self.assertTrue({f["path"]: f for f in report["files"]}["order.py"]["externally_modified"])
        with self.assertRaises(RollbackConflict) as ctx:
            p.runtime([]).rollback(run_id)
        self.assertEqual([pr["path"] for pr in ctx.exception.problems], ["order.py"])
        self.assertEqual(p.read("order.py"), external)         # ни один файл не перезаписан
        self.assertTrue((p.ws / "helper.py").exists())
        self.assertEqual(p.state(run_id)["status"], "completed")

    def test_created_and_deleted_files_are_journaled_and_restored(self):
        p = Project(self, extra_files={"old.txt": "old content"})
        steps = [act(call("write_file", "w1", path="pkg/new.py", content="X = 1\n"),
                     call("delete_file", "d1", path="old.txt")),
                 say("done")]
        run_id = p.runtime(steps).run("t", mode="auto", plan_mode=False)
        p.runtime(steps).approve(run_id)                       # удаление требует подтверждения и в auto
        files = {f["path"]: f for f in p.runtime([]).diff_report(run_id)["files"]}
        self.assertEqual((files["pkg/new.py"]["op"], files["pkg/new.py"]["before"]), ("created", None))
        self.assertEqual((files["old.txt"]["op"], files["old.txt"]["after"]), ("deleted", None))

        result = p.runtime([]).rollback(run_id)
        self.assertEqual(result, {"restored": ["old.txt"], "deleted": ["pkg/new.py"], "unchanged": []})
        self.assertEqual(p.read("old.txt"), "old content")
        self.assertFalse((p.ws / "pkg" / "new.py").exists())

    def test_bash_changes_are_journaled(self):
        p = Project(self, allow=["cp order.py copy.py"])
        run_id = p.runtime([act(call("Bash", "b1", command="cp order.py copy.py", cwd=".")), say("ok")]).run(
            "t", mode="auto", plan_mode=False)
        entries = p.runtime([]).diff_report(run_id)["entries"]
        self.assertEqual([(c["op"], c["path"]) for c in entries[0]["changes"]], [("created", "copy.py")])
        p.runtime([]).rollback(run_id)
        self.assertFalse((p.ws / "copy.py").exists())

    def test_external_change_before_an_action_is_a_conflict(self):
        p = Project(self)
        ex = ToolExecutor()
        run_id = p.runtime(FIX_STEPS).run("fix", mode="safe", plan_mode=False)   # ждёт подтверждения write_file
        (p.ws / "test_order.py").write_text(ORIGINAL_TEST + "# human edit\n")
        rt = p.runtime(FIX_STEPS, executor=ex)
        self.assertEqual(rt.approve(run_id), "waiting")
        st = p.state(run_id)
        self.assertEqual(st["pending"]["kind"], "conflict")
        self.assertEqual([c["path"] for c in st["pending"]["changes"]], ["test_order.py"])
        self.assertEqual(ex.executions, [])
        self.assertEqual(p.read("order.py"), ORIGINAL_ORDER)
        with self.assertRaisesRegex(UserError, "only --changes external"):
            rt.recover(run_id, "retry", changes="attribute")

        self.assertEqual(rt.recover(run_id, "retry", changes="external"), "waiting")   # дальше Bash ждёт подтверждения
        self.assertEqual(p.read("order.py"), FIXED_ORDER)
        files = [f["path"] for f in p.runtime([]).diff_report(run_id)["files"]]
        self.assertEqual(files, ["order.py"], "an external change is not attributed to the run")
        p.runtime(FIX_STEPS).approve(run_id)
        p.runtime([]).rollback(run_id)
        self.assertEqual(p.read("order.py"), ORIGINAL_ORDER)
        self.assertTrue(p.read("test_order.py").endswith("# human edit\n"))

    def test_rollback_waits_until_uncertain_call_is_resolved(self):
        p = Project(self)
        steps = [act(call("write_file", "w1", path="order.py", content=FIXED_ORDER)), say("done")]
        rt = p.runtime(steps, crash_at="after_tool_execute:write_file")
        with self.assertRaises(SimulatedCrash):
            rt.run("fix", mode="auto", plan_mode=False)
        run_id = rt.state["run_id"]

        with self.assertRaisesRegex(UserError, "uncertain"):
            p.runtime(steps).rollback(run_id)                  # сразу после сбоя
        self.assertEqual(p.read("order.py"), FIXED_ORDER)
        self.assertEqual(p.runtime([]).diff_report(run_id)["unresolved_calls"], ["w1"])

        rt = p.runtime(steps)
        rt.resume(run_id)                                       # пользователь видит изменения
        with self.assertRaisesRegex(UserError, "uncertain"):
            rt.rollback(run_id)
        rt.recover(run_id, "fail", changes="attribute")        # изменения — этого запуска
        report = p.runtime([]).diff_report(run_id)
        self.assertEqual([e["kind"] for e in report["entries"]], ["recovered"])
        p.runtime([]).rollback(run_id)
        self.assertEqual(p.read("order.py"), ORIGINAL_ORDER)

    def test_generated_files_are_excluded_explicitly(self):
        p = Project(self)
        run_id = p.runtime(FIX_STEPS).run("fix", mode="safe", plan_mode=False)
        (p.ws / "__pycache__").mkdir()
        (p.ws / "__pycache__" / "order.cpython-310.pyc").write_bytes(b"\x00bytecode")
        self.assertEqual(p.runtime(FIX_STEPS).approve(run_id), "waiting")   # не конфликт: исключено policy
        self.assertEqual(p.state(run_id)["pending"]["kind"], "confirmation")

    def test_second_rollback_is_refused(self):
        p = Project(self)
        run_id = p.runtime(FIX_STEPS).run("fix", mode="auto", plan_mode=False)
        p.runtime([]).rollback(run_id)
        with self.assertRaisesRegex(RollbackError, "already rolled back"):
            p.runtime([]).rollback(run_id)


if __name__ == "__main__":
    unittest.main()
