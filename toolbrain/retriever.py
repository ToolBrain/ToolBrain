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
            topic: Topic that supported by agents of toolbrain
            guideline: Important or specific guideline related to a specific topic

        Returns:
            A dictionary with the same keys, but containing only the most relevant resources

        """
        # Create a prompt for the LLM to select relevant resources
        prompt = """
            You play a role as an expert assistant in the {topic}. Your task is to select the relevant resources to help answer a user's query.

            USER QUERY: {query}

            Below are the available resources. For each category, select items that are directly or indirectly relevant to answering the query.
            Be generous in your selection - include resources that might be useful for the task, even if they're not explicitly mentioned in the query.
            It's better to include slightly more resources than to miss potentially useful ones.

            AVAILABLE TOOLS:
            {tools}

            For each category, respond with ONLY the indices of the relevant items in the following format:
            TOOLS: [list of indices]

            For example:
            TOOLS: [0, 3, 5, 7, 9]

            If a category has no relevant items, use an empty list, e.g., TOOLS: []

            IMPORTANT GUIDELINES:
            {guideline} 
        """.format(topic=topic, 
        query=query, 
        tools=self._format_resources_for_prompt(resources.get("tools", [])),
        guideline=guideline or "1. Focus on relevance to the query.\n2. Be comprehensive in the selection.\n3. Avoid including irrelevant items.")

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
                inputs = resource.get("inputs", None)
                output_type = resource.get("output_type", None)
                formatted.append(f"{i}. {name}: {description}" + f". Inputs: {inputs}. Output_type: {output_type}" if inputs or output_type else "")
            elif isinstance(resource, str):
                # Handle string format (simple strings)
                formatted.append(f"{i}. {resource}")
            else:
                # Try to extract name and description from tool objects
                name = getattr(resource, "name", str(resource))
                desc = getattr(resource, "description", "")
                inputs = getattr(resource, "inputs", None)
                output_type = getattr(resource, "output_type", None)
                formatted.append(f"{i}. Name: {name}, Description: {desc}"+ f". Inputs: {inputs}. Output_type: {output_type}" if inputs or output_type else "")

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