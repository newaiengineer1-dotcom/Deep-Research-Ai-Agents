from ddgs import DDGS


class DuckDuckGoResearchTool:
    """Simple DuckDuckGo wrapper used directly by the research agent."""

    name = "duckduckgo_web_search"

    def __init__(self, max_results=6):
        self.max_results = int(max_results)

    def _run(self, query: str) -> str:
        try:
            results = DDGS().text(
                query,
                region="us-en",
                safesearch="moderate",
                max_results=self.max_results,
            )
            if not results:
                return "No search results were returned."

            output = []
            for i, item in enumerate(results, 1):
                output.append(
                    f"[{i}] {item.get('title', 'Untitled')}\n"
                    f"URL: {item.get('href', '')}\n"
                    f"Snippet: {item.get('body', '')}"
                )
            return "\n\n".join(output)
        except Exception as exc:
            return f"Web search error: {type(exc).__name__}: {exc}"
