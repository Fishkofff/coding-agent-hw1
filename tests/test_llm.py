"""HTTP-клиент OpenAI-compatible API против локального фейкового сервера (без сети и LLM)."""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agent.llm import ModelError, OpenAICompatibleModel
from agent.report import Reporter
from agent.runtime import AgentRuntime
from tests.helpers import FIXED_ORDER, TEST_CMD, Project


def completion(content="", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            for cid, name, args in tool_calls]
    return {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}


class FakeServer:
    """Отдаёт заранее заданные ответы chat/completions и запоминает запросы."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
                status, payload = server.responses.pop(0)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class OpenAICompatibleClientTest(unittest.TestCase):
    def serve(self, responses):
        server = FakeServer(responses)
        self.addCleanup(server.close)
        return server

    def test_agent_runs_against_openai_compatible_api(self):
        server = self.serve([
            (200, completion("reading", [("call_a", "read_file", {"path": "order.py"}),
                                         ("call_b", "read_file", {"path": "test_order.py"})])),
            (200, completion("<think>hidden reasoning</think>fixing", [("call_c", "write_file",
                                                                       {"path": "order.py", "content": FIXED_ORDER})])),
            (200, completion("", [("call_d", "Bash", {"command": TEST_CMD, "cwd": "."})])),
            (200, completion("Done: tests pass.")),
        ])
        p = Project(self)
        model = OpenAICompatibleModel(server.url, "test-key", "fake-model", max_retries=0)
        rt = AgentRuntime(p.policy_path, model, reporter=Reporter(p.out))
        run_id = rt.run("fix the order total", mode="auto", plan_mode=False)
        st = p.state(run_id)
        self.assertEqual(st["status"], "completed")
        self.assertEqual(st["execution_calls"]["call_d"]["result"]["exit_code"], 0)

        first = server.requests[0]
        self.assertEqual(first["path"], "/v1/chat/completions")
        self.assertEqual(first["auth"], "Bearer test-key")
        self.assertEqual(first["body"]["model"], "fake-model")
        bash = next(t for t in first["body"]["tools"] if t["function"]["name"] == "Bash")
        self.assertEqual(bash["type"], "function")
        self.assertEqual(bash["function"]["parameters"]["required"], ["command", "cwd"])

        second = server.requests[1]["body"]["messages"]
        self.assertEqual(second[-3]["tool_calls"][0]["id"], "call_a")
        self.assertEqual([(m["role"], m["tool_call_id"]) for m in second[-2:]],
                         [("tool", "call_a"), ("tool", "call_b")])
        assistant = [m for m in server.requests[2]["body"]["messages"] if m["role"] == "assistant"]
        self.assertEqual(assistant[-1]["content"], "fixing")      # <think> вырезан

    def test_http_error_interrupts_run_and_resume_continues(self):
        server = self.serve([(400, {"error": "bad request"}), (200, completion("ok"))])
        p = Project(self)
        model = OpenAICompatibleModel(server.url, "k", "m", max_retries=0)
        rt = AgentRuntime(p.policy_path, model, reporter=Reporter(p.out))
        run_id = rt.run("t", mode="auto", plan_mode=False)
        st = p.state(run_id)
        self.assertEqual(st["status"], "interrupted")
        self.assertIn("HTTP 400", st["error"]["message"])
        self.assertEqual(AgentRuntime(p.policy_path, model, reporter=Reporter(p.out)).resume(run_id), "completed")
        self.assertEqual(p.state(run_id)["budget"]["used"]["model_calls"], 2)

    def test_parse_handles_dict_arguments_and_missing_ids(self):
        response = OpenAICompatibleModel.parse({"choices": [{"message": {"content": None, "tool_calls": [
            {"function": {"name": "read_file", "arguments": {"path": "order.py"}}}]}}]})
        self.assertEqual(response.tool_calls[0].arguments, '{"path": "order.py"}')
        self.assertEqual(response.tool_calls[0].id, "")
        with self.assertRaises(ModelError):
            OpenAICompatibleModel.parse({"unexpected": True})


if __name__ == "__main__":
    unittest.main()
