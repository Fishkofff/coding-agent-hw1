"""Блок 1: agent loop, tool calling, Bash, ошибки инструментов."""
import time
import unittest

from agent.llm import ToolCallRequest
from agent.tools import REGISTRY, ToolExecutor
from tests.helpers import (FIXED_ORDER, ORIGINAL_ORDER, TEST_CMD, Project, act, call, say, tool_messages,
                           tool_results)


class AgentLoopTest(unittest.TestCase):
    def test_control_task_is_fixed_and_verified_with_bash(self):
        p = Project(self)
        test_sha = p.sha("test_order.py")
        rt = p.runtime([
            act(call("list_files", "c1", path=".")),
            act(call("read_file", "c2", path="order.py"), call("read_file", "c3", path="test_order.py")),
            act(call("write_file", "c4", path="order.py", content=FIXED_ORDER)),
            act(call("Bash", "c5", command=TEST_CMD, cwd=".")),
            say("Исправлено, тесты проходят."),
        ])
        run_id = rt.run("Исправь расчёт итоговой стоимости заказа, чтобы тесты проходили. Тесты не меняй.",
                        mode="auto", plan_mode=False)
        st = p.state(run_id)
        self.assertEqual(st["status"], "completed")
        self.assertEqual(st["final_answer"], "Исправлено, тесты проходят.")
        bash = st["execution_calls"]["c5"]
        self.assertEqual(bash["name"], "Bash")
        self.assertEqual(bash["result"]["exit_code"], 0)
        self.assertIn("Ran 2 tests", bash["result"]["stderr"])
        self.assertIn("OK", bash["result"]["stderr"])
        self.assertEqual(p.read("order.py"), FIXED_ORDER)
        self.assertEqual(p.sha("test_order.py"), test_sha, "test_order.py must stay unchanged")
        # Каждый tool call получил ответ role=tool с исходным tool_call_id, в исходном порядке.
        self.assertEqual([cid for cid, _ in tool_messages(st)], ["c1", "c2", "c3", "c4", "c5"])
        # Два вызова из одного ответа модели лежат в одном assistant-сообщении.
        multi = [m for m in st["messages"] if m["role"] == "assistant" and len(m.get("tool_calls", [])) == 2]
        self.assertEqual([c["id"] for c in multi[0]["tool_calls"]], ["c2", "c3"])

    def test_tools_are_passed_with_name_description_and_json_schema(self):
        p = Project(self)
        rt = p.runtime([say("nothing to do")])
        rt.run("t", mode="auto", plan_mode=False)
        self.assertEqual(rt.model.requests[0]["tools"], ["Bash", "list_files", "read_file", "write_file", "delete_file"])
        for spec in REGISTRY.values():
            function = spec.to_openai()["function"]
            self.assertTrue(function["name"] and function["description"])
            self.assertEqual(function["parameters"]["type"], "object")
        self.assertEqual(REGISTRY["Bash"].parameters["required"], ["command", "cwd"])

    def test_tool_errors_are_returned_to_the_model_and_do_not_crash_the_run(self):
        p = Project(self)
        seen = []

        def after_errors(messages, tools):
            seen.extend(m for m in messages if m["role"] == "tool")
            return act(call("read_file", "ok1", path="order.py"))

        rt = p.runtime([
            act(call("read_file", "e1", path="missing.py"),
                call("no_such_tool", "e2", x=1),
                call("write_file", "e3", path="order.py"),            # нет обязательного content
                ToolCallRequest("e4", "read_file", "{not json"),
                call("read_file", "e5", path="order.py", extra=1)),   # лишнее поле
            after_errors,
            say("done"),
        ])
        run_id = rt.run("t", mode="auto", plan_mode=False)
        st = p.state(run_id)
        results = tool_results(st)
        self.assertEqual(st["status"], "completed")
        self.assertEqual(results["e1"]["error"], "invalid_arguments")
        self.assertIn("does not exist", results["e1"]["message"])
        self.assertEqual(results["e2"]["error"], "unknown_tool")
        self.assertEqual(results["e3"]["error"], "invalid_arguments")
        self.assertIn("content", results["e3"]["message"])
        self.assertEqual(results["e4"]["error"], "invalid_arguments")
        self.assertIn("unexpected property 'extra'", results["e5"]["message"])
        self.assertEqual(len(seen), 5, "the model saw every error before choosing the next step")
        self.assertTrue(results["ok1"]["ok"])
        self.assertEqual(p.read("order.py"), ORIGINAL_ORDER)

    def test_bash_has_timeout_and_output_limit(self):
        p = Project(self, allow=["sleep 5", "cat big.txt"], bash={"timeout_sec": 0.5, "max_output_bytes": 100},
                    extra_files={"big.txt": "x" * 5000})
        rt = p.runtime([act(call("Bash", "b1", command="sleep 5", cwd="."),
                            call("Bash", "b2", command="cat big.txt", cwd=".")), say("ok")])
        started = time.monotonic()
        run_id = rt.run("t", mode="auto", plan_mode=False)
        self.assertLess(time.monotonic() - started, 4)
        results = tool_results(p.state(run_id))
        self.assertTrue(results["b1"]["timed_out"])
        self.assertFalse(results["b1"]["ok"])
        self.assertTrue(results["b2"]["truncated"])
        self.assertLessEqual(len(results["b2"]["stdout"].encode()), 100)
        self.assertEqual(results["b2"]["output_bytes_total"], 5000)

    def test_bash_gets_no_agent_secrets(self):
        p = Project(self, allow=["env"])
        import os
        os.environ["OPENAI_API_KEY"] = "sk-test-secret"
        self.addCleanup(os.environ.pop, "OPENAI_API_KEY", None)
        run_id = p.runtime([act(call("Bash", "b1", command="env", cwd=".")), say("ok")]).run(
            "t", mode="auto", plan_mode=False)
        stdout = tool_results(p.state(run_id))["b1"]["stdout"]
        self.assertNotIn("sk-test-secret", stdout)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", stdout)

    def test_missing_tool_call_id_gets_a_synthetic_one(self):
        p = Project(self)
        ex = ToolExecutor()
        run_id = p.runtime([act(ToolCallRequest("", "read_file", '{"path": "order.py"}')), say("ok")],
                           executor=ex).run("t", mode="auto", plan_mode=False)
        st = p.state(run_id)
        (call_id, result), = tool_messages(st)
        self.assertTrue(call_id.startswith("rt-"))
        self.assertTrue(result["ok"])


if __name__ == "__main__":
    unittest.main()
