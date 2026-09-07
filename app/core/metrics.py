"""Small dependency-free Prometheus exposition registry.

Labels are intentionally finite (method, route template, status class, tool,
outcome) so user input cannot create unbounded metric cardinality.
"""
from collections import defaultdict
from threading import Lock

_lock = Lock()
_http_requests: dict[tuple[str, str, str], int] = defaultdict(int)
_http_duration: dict[tuple[str, str], tuple[int, float]] = defaultdict(lambda: (0, 0.0))
_commerce_calls: dict[tuple[str, str], int] = defaultdict(int)
_commerce_duration: dict[tuple[str, str], tuple[int, float]] = defaultdict(lambda: (0, 0.0))


def record_http(method: str, route: str, status_code: int, duration_seconds: float) -> None:
    status = f"{status_code // 100}xx"
    with _lock:
        _http_requests[(method, route, status)] += 1
        count, total = _http_duration[(method, route)]
        _http_duration[(method, route)] = (count + 1, total + max(duration_seconds, 0.0))


def record_commerce_call(tool: str, outcome: str, duration_seconds: float) -> None:
    with _lock:
        _commerce_calls[(tool, outcome)] += 1
        count, total = _commerce_duration[(tool, outcome)]
        _commerce_duration[(tool, outcome)] = (count + 1, total + max(duration_seconds, 0.0))


def _labels(**values: str) -> str:
    return ",".join(f'{key}="{value}"' for key, value in values.items())


def render_prometheus() -> str:
    lines = [
        "# HELP agent_http_requests_total HTTP requests handled by the Agent runtime.",
        "# TYPE agent_http_requests_total counter",
    ]
    with _lock:
        for (method, route, status), count in sorted(_http_requests.items()):
            lines.append(f"agent_http_requests_total{{{_labels(method=method, route=route, status=status)}}} {count}")
        lines += [
            "# HELP agent_http_request_duration_seconds_total Total HTTP request duration in seconds.",
            "# TYPE agent_http_request_duration_seconds_total counter",
        ]
        for (method, route), (count, total) in sorted(_http_duration.items()):
            labels = _labels(method=method, route=route)
            lines.append(f"agent_http_request_duration_seconds_total{{{labels}}} {total:.9f}")
            lines.append(f"agent_http_request_duration_seconds_count{{{labels}}} {count}")
        lines += [
            "# HELP agent_commerce_tool_calls_total Allow-listed Java Commerce tool calls.",
            "# TYPE agent_commerce_tool_calls_total counter",
        ]
        for (tool, outcome), count in sorted(_commerce_calls.items()):
            lines.append(f"agent_commerce_tool_calls_total{{{_labels(tool=tool, outcome=outcome)}}} {count}")
        lines += [
            "# HELP agent_commerce_tool_duration_seconds_total Total allow-listed Commerce tool duration.",
            "# TYPE agent_commerce_tool_duration_seconds_total counter",
        ]
        for (tool, outcome), (count, total) in sorted(_commerce_duration.items()):
            labels = _labels(tool=tool, outcome=outcome)
            lines.append(f"agent_commerce_tool_duration_seconds_total{{{labels}}} {total:.9f}")
            lines.append(f"agent_commerce_tool_duration_seconds_count{{{labels}}} {count}")
    return "\n".join(lines) + "\n"
