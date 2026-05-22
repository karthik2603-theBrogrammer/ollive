from api.agent_tools.registry import get_api_tools

__all__ = ["get_tools"]


def get_tools():
    return get_api_tools()
