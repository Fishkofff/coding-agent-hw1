"""Уровень 8: единый бюджет запуска, устойчивый к сбоям."""
import unittest

from agent.faults import SimulatedCrash
from agent.tools import ToolExecutor
from tests.helpers import FIXED_ORDER, TEST_CMD, Project, act, call, say, tool_results

BIG = {"model_calls": 50, "tool_calls": 100, "side_effects": 50, "result_bytes": 500000}


def budget(**limits):
    return {**BIG, **limits}


class BudgetLimitsTest(unittest.TestCase):
    def test_model_calls_limit(self):
        p = Project(self, budget=budget(model_calls=2))
        steps = [act(call("read_file", "r1", path="order.py")), act(call("read_file", "r2", path="order.py")),
                 say("never reached")]
        rt = p.runtime(steps)
        run_id = rt.run("t", mode="auto", plan_mode=False)
        st = p.state(run_id)
        self.assertEqual(st["status"], "failed")
        self.assertEqual(st["error"]["code"], "budget_exhausted")
        self.assertIn("model_calls", st["error"]["message"])
        self.assertEqual(len(rt.model.requests), 2, "no request is made once the limit is exhausted")
        self.assertEqual(st["budget"]["used"]["model_calls"], 2)

    def test_tool_calls_limit_rejects_the_whole_batch(self):
        p = Project(self, budget=budget(tool_calls=3))
        ex = ToolExecutor()
        steps = [act(call("read_file", "r1", path="order.py"), call("read_file", "r2", path="test_order.py")),
                 act(call("read_file", "r3", path="order.py"), call("read_file", "r4", path="test_order.py")),
                 say("done")]
        run_id = p.runtime(steps, executor=ex).run("t", mode="auto", plan_mode=False)
        st = p.state(run_id)
        results = tool_results(st)
        self.assertEqual(len(ex.executions), 2)                 # из второй пачки (2 > 1 остатка) — ни одного
        self.assertEqual(results["r3"]["error"], "budget_exceeded")
        self.assertEqual(results["r4"]["error"], "budget_exceeded")
        self.assertEqual(st["budget"]["used"]["tool_calls"], 2)

    def test_side_effects_limit(self):
        p = Project(self, budget=budget(side_effects=1))
        ex = ToolExecutor()
        steps = [act(call("write_file", "w1", path="a.txt", content="a"),
                     call("write_file", "w2", path="b.txt", content="b")),   # 2 побочных эффекта > 1: пачка целиком отклонена
                 act(call("write_file", "w3", path="a.txt", content="a")),   # помещается
                 act(call("write_file", "w4", path="b.txt", content="b")),   # лимит исчерпан
                 say("done")]
        run_id = p.runtime(steps, executor=ex).run("t", mode="auto", plan_mode=False)
        st = p.state(run_id)
        results = tool_results(st)
        self.assertEqual([args["path"] for _, args in ex.executions], ["a.txt"])
        self.assertEqual(results["w1"]["error"], "budget_exceeded")
        self.assertEqual(results["w2"]["error"], "budget_exceeded")
        self.assertTrue(results["w3"]["ok"])
        self.assertEqual(results["w4"]["error"], "budget_exceeded")
        self.assertFalse((p.ws / "b.txt").exists())
        self.assertEqual(st["budget"]["used"]["side_effects"], 1)

    def test_result_bytes_limit_truncates_and_marks_output_incomplete(self):
        p = Project(self, budget=budget(result_bytes=600), extra_files={"big.txt": "line\n" * 1000})
        ex = ToolExecutor()
        steps = [act(call("read_file", "r1", path="big.txt")), act(call("read_file", "r2", path="order.py")),
                 say("done")]
        run_id = p.runtime(steps, executor=ex).run("t", mode="auto", plan_mode=False)
        st = p.state(run_id)
        results = tool_results(st)
        self.assertTrue(results["r1"]["truncated"])
        self.assertIn("truncated", results["r1"]["content"])
        self.assertLessEqual(st["budget"]["used"]["result_bytes"], 600)
        self.assertEqual(results["r2"]["error"], "budget_exceeded")   # остаток меньше минимума — не запускается
        self.assertEqual(len(ex.executions), 1)

    def test_bash_receives_only_the_remaining_result_budget(self):
        p = Project(self, allow=["cat big.txt"], budget=budget(result_bytes=700), extra_files={"big.txt": "y" * 50_000})
        run_id = p.runtime([act(call("Bash", "b1", command="cat big.txt", cwd=".")), say("ok")]).run(
            "t", mode="auto", plan_mode=False)
        st = p.state(run_id)
        result = tool_results(st)["b1"]
        self.assertTrue(result["truncated"])
        self.assertEqual(result["exit_code"], 0)
        self.assertLessEqual(st["budget"]["used"]["result_bytes"], 700)


class BudgetPersistenceTest(unittest.TestCase):
    def test_counters_survive_plan_approval_and_resume(self):
        p = Project(self)
        plan = {"goal": "g", "steps": [{"id": "T1", "description": "fix", "depends_on": [], "tools": ["write_file"],
                                        "checks": [{"type": "file_changed", "path": "order.py"}]}],
                "risks": [], "assumptions": [], "done_criteria": "d"}
        steps = [act(call("read_file", "p0", path="order.py")), act(call("submit_plan", "p1", **plan)),
                 act(call("write_file", "w1", path="order.py", content=FIXED_ORDER)),
                 act(call("complete_task", "c1", task_id="T1")), say("done")]
        run_id = p.runtime(steps).run("fix", mode="safe", plan_mode=True)
        used_planning = dict(p.state(run_id)["budget"]["used"])
        self.assertEqual((used_planning["model_calls"], used_planning["tool_calls"]), (2, 2))

        p.runtime(steps).approve_plan(run_id)                 # другой экземпляр runtime
        used_after_plan = dict(p.state(run_id)["budget"]["used"])
        self.assertEqual(used_after_plan["model_calls"], 3)
        self.assertEqual(used_after_plan["tool_calls"], 3)

        p.runtime(steps).approve(run_id)
        used_final = p.state(run_id)["budget"]["used"]
        self.assertEqual(used_final["model_calls"], 5)
        self.assertEqual(used_final["tool_calls"], 4)
        self.assertEqual(used_final["side_effects"], 1)
        self.assertGreaterEqual(used_final["result_bytes"], used_after_plan["result_bytes"])

    def test_crash_after_charge_does_not_refund_budget(self):
        p = Project(self)
        steps = [act(call("write_file", "w1", path="order.py", content=FIXED_ORDER)), say("done")]
        rt = p.runtime(steps, crash_at="after_tool_execute:write_file")
        with self.assertRaises(SimulatedCrash):
            rt.run("t", mode="auto", plan_mode=False)
        run_id = rt.state["run_id"]
        used = p.state(run_id)["budget"]["used"]
        self.assertEqual((used["model_calls"], used["tool_calls"], used["side_effects"]), (1, 1, 1))
        self.assertEqual(used["result_bytes"], 0)              # результат не сохранён — не доставлен модели

        rt = p.runtime(steps)
        rt.resume(run_id)
        self.assertEqual(p.state(run_id)["budget"]["used"]["side_effects"], 1)   # resume не обнуляет
        rt.recover(run_id, "retry", changes="attribute")
        used = p.state(run_id)["budget"]["used"]
        self.assertEqual((used["tool_calls"], used["side_effects"]), (2, 2))     # повтор — новая попытка, новое списание

    def test_crash_after_model_call_was_charged(self):
        p = Project(self)
        steps = [act(call("read_file", "r1", path="order.py")), say("done")]
        rt = p.runtime(steps, crash_at="after_model_charge")
        with self.assertRaises(SimulatedCrash):
            rt.run("t", mode="auto", plan_mode=False)
        run_id = rt.state["run_id"]
        self.assertEqual(p.state(run_id)["budget"]["used"]["model_calls"], 1)
        self.assertEqual(p.runtime(steps).resume(run_id), "completed")
        self.assertEqual(p.state(run_id)["budget"]["used"]["model_calls"], 3)   # потерянный запрос не возвращён

    def test_retry_is_refused_when_budget_is_exhausted(self):
        p = Project(self, budget=budget(side_effects=1))
        steps = [act(call("write_file", "w1", path="order.py", content=FIXED_ORDER)), say("done")]
        rt = p.runtime(steps, crash_at="after_tool_execute:write_file")
        with self.assertRaises(SimulatedCrash):
            rt.run("t", mode="auto", plan_mode=False)
        run_id = rt.state["run_id"]
        ex = ToolExecutor()
        rt = p.runtime(steps, executor=ex)
        rt.resume(run_id)
        self.assertEqual(rt.recover(run_id, "retry", changes="attribute"), "completed")
        self.assertEqual(ex.executions, [])
        self.assertEqual(tool_results(p.state(run_id))["w1"]["error"], "budget_exceeded")


if __name__ == "__main__":
    unittest.main()
