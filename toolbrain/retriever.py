import contextlib
import re
import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


class ToolRetriever:
    """Retrieve tools from the tool registry."""

    def __init__(self):
        pass

    def prompt_based_retrieval(self, query: str, resources: dict, llm=None, topic='bio medial', guideline="") -> dict:
        """Use a prompt-based approach to retrieve the most relevant resources for a query.

        Args:
            query: The user's query
            resources: A dictionary with keys 'tools', 'data_lake', and 'libraries',
                      each containing a list of available resources
            llm: Optional LLM instance to use for retrieval (if None, will create a new one)

        Returns:
            A dictionary with the same keys, but containing only the most relevant resources

        """
        # Create a prompt for the LLM to select relevant resources
        prompt = f"""
You play a role as an expert assistant in the {topic}. Your task is to select the relevant resources to help answer a user's query.

USER QUERY: {query}

Below are the available resources. For each category, select items that are directly or indirectly relevant to answering the query.
Be generous in your selection - include resources that might be useful for the task, even if they're not explicitly mentioned in the query.
It's better to include slightly more resources than to miss potentially useful ones.

AVAILABLE TOOLS:
{self._format_resources_for_prompt(resources.get("tools", []))}

For each category, respond with ONLY the indices of the relevant items in the following format:
TOOLS: [list of indices]

For example:
TOOLS: [0, 3, 5, 7, 9]
DATA_LAKE: [1, 2, 4]
LIBRARIES: [0, 2, 4, 5, 8]

If a category has no relevant items, use an empty list, e.g., TOOLS: []

IMPORTANT GUIDELINES:
{guideline if guideline else "1. Focus on relevance to the query.\n2. Be comprehensive in your selection.\n3. Avoid including irrelevant items."}
"""

        # Use the provided LLM or create a new one
        if llm is None:
            llm = ChatOpenAI(model="gpt-4o")

        # Invoke the LLM
        if hasattr(llm, "invoke"):
            # For LangChain-style LLMs
            response = llm.invoke([HumanMessage(content=prompt)])
            response_content = response.content
        else:
            # For other LLM interfaces
            response_content = str(llm(prompt))

        # Parse the response to extract the selected indices
        selected_indices = self._parse_llm_response(response_content)

        # Get the selected resources
        selected_resources = {
            "tools": [
                resources["tools"][i] for i in selected_indices.get("tools", []) if i < len(resources.get("tools", []))
            ],
        }

        return selected_resources

    def _format_resources_for_prompt(self, resources: list) -> str:
        """Format resources for inclusion in the prompt."""
        formatted = []
        for i, resource in enumerate(resources):
            if isinstance(resource, dict):
                # Handle dictionary format (from tool registry or data lake/libraries with descriptions)
                name = resource.get("name", f"Resource {i}")
                description = resource.get("description", "")
                formatted.append(f"{i}. {name}: {description}")
            elif isinstance(resource, str):
                # Handle string format (simple strings)
                formatted.append(f"{i}. {resource}")
            else:
                # Try to extract name and description from tool objects
                name = getattr(resource, "name", str(resource))
                desc = getattr(resource, "description", "")
                formatted.append(f"{i}. {name}: {desc}")

        return "\n".join(formatted) if formatted else "None available"

    def _parse_llm_response(self, response: str) -> dict:
        """Parse the LLM response to extract the selected indices."""
        selected_indices = {"tools": []}

        # Extract indices for each category
        tools_match = re.search(r"TOOLS:\s*\[(.*?)\]", response, re.IGNORECASE)
        if tools_match and tools_match.group(1).strip():
            with contextlib.suppress(ValueError):
                selected_indices["tools"] = [int(idx.strip()) for idx in tools_match.group(1).split(",") if idx.strip()]

        return selected_indices

if __name__ == "__main__":
    # Example usage
    retriever = ToolRetriever()
    api_key = os.getenv("OPEN_AI_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    llm = ChatOpenAI(model="gpt-4o", api_key=api_key)  # Replace with your actual API key
    guideline = """
        1. Be generous but not excessive - aim to include all potentially relevant resources
        2. ALWAYS prioritize database tools for general queries - include as many database tools as possible
        3. Include all literature search tools
        4. For wet lab sequence type of queries, ALWAYS include molecular biology tools
        5. For data lake items, include datasets that could provide useful information
        6. For libraries, include those that provide functions needed for analysis
        7. Don't exclude resources just because they're not explicitly mentioned in the query
        8. When in doubt about a database tool or molecular biology tool, include it rather than exclude it
    """
    resources = {
        "tools": [
            {"name": "BLAST", "description": "Tool for comparing an amino acid or nucleotide sequence to sequence databases."},
            {"name": "ClustalW", "description": "Tool for multiple sequence alignment."},
            {"name": "Gene Ontology", "description": "A framework for the model of biology that relates to gene functions."},
            {"name": "OpenCV", "description": "A framework for computer vision tasks."},
            {"name": "ReactJS", "description": "A framework for web development."},
        ],
    }
    list_pairs = [("Find similar protein sequences and analyze their functions.", "bio medical"), ("Blur region of a cat in my uploaded image and show it at top right of my website.", "photo editing"), ("Blur region of a cat in my uploaded image.", "photo editing")]
    for query, topic in list_pairs:
        print(f"\nQuery: {query}\nTopic: {topic}")
        selected = retriever.prompt_based_retrieval(query, resources, topic=topic, llm=llm, guideline=guideline)
        print("Selected Resources:", selected)