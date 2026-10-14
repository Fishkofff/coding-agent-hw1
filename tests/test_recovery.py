"""Блок 3: восстановление после сбоя, неопределённые вызовы, повтор tool_call_id."""
import unittest

from agent.faults import SimulatedCrash
from agent.runtime import UserError
from agent.tools import ToolExecutor
from tests.helpers import (FIXED_ORDER, ORIGINAL_ORDER, TEST_CMD, Project, act, call, say, tool_messages,
                           tool_results)

FIX_STEPS = [act(call("write_file", "w1", path="order.py", content=FIXED_ORDER)),
             act(call("Bash", "b1", command=TEST_CMD, cwd=".")),
             say("done")]


def crash(testcase, project, steps, point, **run_kwargs):
    """Запуск падает в управляемой точке; возвращает run_id. Состояние в памяти теряется, на диске — остаётся."""
    rt = project.runtime(steps, crash_at=point)
    with testcase.assertRaises(SimulatedCrash):
        rt.run("fix", **{"mode": "auto", "plan_mode": False, **run_kwargs})
    return rt.state["run_id"]


class CrashRecoveryTest(unittest.TestCase):
    def test_crash_between_execution_and_saving_result_makes_call_uncertain(self):
        p = Project(self)
        run_id = crash(self, p, FIX_STEPS, "after_tool_execute:write_file")
        st = p.state(run_id)
        record = st["execution_calls"]["w1"]
        self.assertEqual(record["status"], "started")          # на диске: «началось», результата нет
        self.assertIsNone(record["result"])
        self.assertEqual(p.read("order.py"), FIXED_ORDER)      # но действие успело выполниться

        ex = ToolExecutor()
        rt = p.runtime(FIX_STEPS, executor=ex)                 # перезапуск
        self.assertEqual(rt.resume(run_id), "waiting")
        st = p.state(run_id)
        self.assertEqual(st["pending"]["kind"], "recovery")
        self.assertEqual(st["execution_calls"]["w1"]["status"], "uncertain")
        self.assertEqual([(c["op"], c["path"]) for c in st["pending"]["changes"]], [("modified", "order.py")])
        self.assertEqual(ex.executions, [], "uncertain call must not be silently re-executed")
        self.assertEqual(rt.model.requests, [], "the model is not asked before the user decides")
        self.assertIn("Неопределённый вызов", p.output)

    def test_retry_is_a_new_attempt_of_the_same_tool_call(self):
        p = Project(self)
        run_id = crash(self, p, FIX_STEPS, "after_tool_execute:write_file")
        ex = ToolExecutor()
        rt = p.runtime(FIX_STEPS, executor=ex)
        rt.resume(run_id)
        with self.assertRaisesRegex(UserError, "--changes"):    # обнаружены изменения — решить, чьи они
            rt.recover(run_id, "retry")
        self.assertEqual(rt.recover(run_id, "retry", changes="attribute"), "completed")
        record = p.state(run_id)["execution_calls"]["w1"]
        self.assertEqual([a["n"] for a in record["attempts"]], [1, 2])
        self.assertEqual([a["status"] for a in record["attempts"]], ["uncertain_retried", "completed"])
        self.assertEqual([tool for tool, _ in ex.executions], ["write_file", "Bash"])

    def test_mark_failed_and_continue(self):
        p = Project(self)
        run_id = crash(self, p, FIX_STEPS, "after_tool_execute:write_file")
        rt = p.runtime(FIX_STEPS)
        rt.resume(run_id)
        self.assertEqual(rt.recover(run_id, "fail", changes="attribute"), "completed")
        st = p.state(run_id)
        self.assertEqual(st["execution_calls"]["w1"]["status"], "failed")
        self.assertEqual(tool_results(st)["w1"]["error"], "outcome_unknown")

    def test_abort_run(self):
        p = Project(self)
        run_id = crash(self, p, FIX_STEPS, "after_tool_execute:write_file")
        rt = p.runtime(FIX_STEPS)
        rt.resume(run_id)
        self.assertEqual(rt.recover(run_id, "abort", changes="external"), "failed")
        self.assertEqual(p.state(run_id)["error"]["code"], "aborted_after_recovery")

    def test_bash_is_not_safely_retryable(self):
        p = Project(self)
        run_id = crash(self, p, FIX_STEPS, "after_tool_execute:Bash")
        rt = p.runtime(FIX_STEPS)
        rt.resume(run_id)
        with self.assertRaisesRegex(UserError, "not safely retryable"):
            rt.recover(run_id, "retry")
        self.assertEqual(rt.recover(run_id, "fail"), "completed")   # тесты не меняют файлы -> --changes не нужен

    def test_completed_calls_are_not_repeated_after_restart(self):
        p = Project(self)
        steps = [act(call("write_file", "w1", path="new1.txt", content="1"),
                     call("write_file", "w2", path="new2.txt", content="2")),
                 say("done")]
        run_id = crash(self, p, steps, "after_result_save:write_file")   # упали после сохранения w1
        st = p.state(run_id)
        self.assertEqual(st["execution_calls"]["w1"]["status"], "completed")
        self.assertEqual(st["execution_calls"]["w2"]["status"], "new")
        ex = ToolExecutor()
        self.assertEqual(p.runtime(steps, executor=ex).resume(run_id), "completed")
        self.assertEqual([args["path"] for _, args in ex.executions], ["new2.txt"])
        self.assertEqual([cid for cid, _ in tool_messages(p.state(run_id))], ["w1", "w2"])

    def test_interrupted_read_is_safely_repeated(self):
        p = Project(self)
        steps = [act(call("read_file", "r1", path="order.py")), say("done")]
        run_id = crash(self, p, steps, "after_tool_execute:read_file")
        self.assertEqual(p.runtime(steps).resume(run_id), "completed")
        record = p.state(run_id)["execution_calls"]["r1"]
        self.assertEqual([a["status"] for a in record["attempts"]], ["interrupted", "completed"])

    def test_history_and_decisions_survive_restart(self):
        p = Project(self)
        steps = [act(call("read_file", "r1", path="order.py")),
                 act(call("write_file", "w1", path="order.py", content=FIXED_ORDER)), say("done")]
        run_id = p.runtime(steps).run("fix", mode="safe", plan_mode=False)
        before = p.state(run_id)
        p.runtime(steps).approve(run_id)
        after = p.state(run_id)
        self.assertEqual(after["messages"][:len(before["messages"])], before["messages"])
        self.assertEqual(after["execution_calls"]["w1"]["decision"]["by"], "user")
        self.assertEqual(after["status"], "completed")


class ToolCallIdempotencyTest(unittest.TestCase):
    def test_repeated_tool_call_id_returns_stored_result_without_reexecution(self):
        p = Project(self)
        steps = [act(call("write_file", "w1", path="order.py", content=FIXED_ORDER)),
                 act(call("write_file", "w1", path="order.py", content=FIXED_ORDER)),   # тот же id и аргументы
                 say("done")]
        ex = ToolExecutor()
        run_id = p.runtime(steps, executor=ex).run("t", mode="auto", plan_mode=False)
        st = p.state(run_id)
        self.assertEqual(len(ex.executions), 1)
        self.assertEqual(st["budget"]["used"]["side_effects"], 1)
        self.assertEqual(len([e for e in st["journal"]["entries"] if e["kind"] == "action"]), 1)
        (_, first), (_, second) = tool_messages(st)
        self.assertTrue(first["ok"])
        self.assertTrue(second["replayed"])
        self.assertEqual(second["original_result"], first)

    def test_repeated_tool_call_id_with_other_arguments_is_a_protocol_error(self):
        p = Project(self)
        steps = [act(call("write_file", "w1", path="order.py", content=FIXED_ORDER)),
                 act(call("write_file", "w1", path="order.py", content="evil = True\n")),
                 say("done")]
        ex = ToolExecutor()
        run_id = p.runtime(steps, executor=ex).run("t", mode="auto", plan_mode=False)
        (_, _first), (_, second) = tool_messages(p.state(run_id))
        self.assertEqual(second["error"], "protocol_error")
        self.assertEqual(len(ex.executions), 1)
        self.assertEqual(p.read("order.py"), FIXED_ORDER)

    def test_same_id_twice_in_one_response_is_a_protocol_error(self):
        p = Project(self)
        ex = ToolExecutor()
        steps = [act(call("read_file", "x", path="order.py"), call("read_file", "x", path="test_order.py")),
                 say("done")]
        run_id = p.runtime(steps, executor=ex).run("t", mode="auto", plan_mode=False)
        (_, first), (_, second) = tool_messages(p.state(run_id))
        self.assertTrue(first["ok"])
        self.assertEqual(second["error"], "protocol_error")
        self.assertEqual(p.read("order.py"), ORIGINAL_ORDER)


if __name__ == "__main__":
    unittest.main()
