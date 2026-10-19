"""DeepSeek Harness, ported to the SGLang Workbench.

Everything is a plugin: the session log, the prompt assembly, the tool registry,
the model adapter, the agent loop and compaction are all rows in one composition
(`dsh.profiles`), and every harness component is mounted beside them.
"""

from .kernel import Context, Plugin, PluginHost
from .loop import Agent, AgentLoop
from .prompt import SystemPrompt
from .session import Session, SessionStore
from .tools import Tool, ToolRuntime

__all__ = [
    "Context", "Plugin", "PluginHost", "Agent", "AgentLoop", "SystemPrompt",
    "Session", "SessionStore", "Tool", "ToolRuntime",
]
__version__ = "1.0.0"
