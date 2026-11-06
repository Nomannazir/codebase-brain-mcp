import pytest
from mcp import Client

from project_brain_mcp.server import build_server

pytestmark = pytest.mark.anyio


async def test_tools_are_listed_with_read_only_hints(brain):
    async with Client(build_server(brain)) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}

    assert set(tools) == {
        "index_repo",
        "search",
        "ask_codebase",
        "read_lines",
        "list_repos",
        "explain_history",
        "search_history",
        "impact_of_change",
        "find_doc_drift",
    }
    assert tools["search"].annotations.read_only_hint is True
    assert tools["index_repo"].annotations.read_only_hint is False


async def test_index_then_ask_returns_cited_structured_sources(brain, sample_repo):
    async with Client(build_server(brain)) as client:
        indexed = await client.call_tool("index_repo", {"source": str(sample_repo)})
        asked = await client.call_tool("ask_codebase", {"question": "How are sessions created?"})

    assert not indexed.is_error and indexed.structured_content["files_seen"] == 3
    sources = asked.structured_content["sources"]
    assert sources and all(":" in s["citation"] for s in sources)
    assert any(s["symbol"] == "SessionStore" for s in sources)


async def test_errors_come_back_as_tool_errors(brain):
    async with Client(build_server(brain)) as client:
        result = await client.call_tool("search", {"query": "anything"})

    assert result.is_error
    assert "Nothing is indexed" in result.content[0].text


async def test_prompt_and_resource(brain, sample_repo):
    async with Client(build_server(brain)) as client:
        await client.call_tool("index_repo", {"source": str(sample_repo)})
        prompt = await client.get_prompt("answer_with_citations", {"question": "What is billed?"})
        resource = await client.read_resource("brain://repos")

    assert "ask_codebase" in prompt.messages[0].content.text
    assert "sample-app" in resource.contents[0].text


async def test_explain_history_tool_returns_structured_changes(brain, git_repo):
    async with Client(build_server(brain)) as client:
        await client.call_tool("index_repo", {"source": str(git_repo)})
        result = await client.call_tool("explain_history", {"symbol": "discount"})
        asked = await client.call_tool("ask_codebase", {"question": "why are discounts rounded?"})

    data = result.structured_content
    assert data["citation"] == "pricing.py:1-2"
    assert len(data["changes"]) == 3 and len(data["changes"][0]["commit"]["sha"]) == 10
    assert asked.structured_content["related_commits"][0]["pull_requests"] == [12]


async def test_impact_and_drift_tools(brain, sample_repo):
    (sample_repo / "README.md").write_text("# App\n\nUse `hash_passwords()` to hash.\n")
    async with Client(build_server(brain)) as client:
        await client.call_tool("index_repo", {"source": str(sample_repo)})
        impact = await client.call_tool("impact_of_change", {"symbol": "hash_password"})
        drift = await client.call_tool("find_doc_drift", {})

    assert impact.structured_content["symbol"] == "hash_password"
    findings = drift.structured_content["findings"]
    assert findings[0]["claim"] == "`hash_passwords()`" and "hash_password" in findings[0]["suggestion"]
