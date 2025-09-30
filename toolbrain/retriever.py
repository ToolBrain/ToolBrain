import contextlib
import re
import os
import traceback

# from langchain_core.messages import HumanMessage
# from langchain_openai import ChatOpenAI
from openai import OpenAI   




class ToolRetriever:
    """Retrieve tools from the tool registry."""

    def __init__(self, llm_model="gpt-4o-mini",
                 llm_instance=None,
                 retrieval_topic: str = "general",  # Topic for tool retrieval
                 retrieval_guidelines: str = "1. Focus on relevance to the query.\n2. Be comprehensive in the selection.\n3. Avoid including irrelevant items.",  # Custom guidelines for tool selection
                 ):
        self.llm_model = llm_model
        self.llm_instance = llm_instance
        self.retrieval_topic = retrieval_topic
        self.retrieval_guidelines = retrieval_guidelines
        # pass

    def select_relevant_tools(self, query: str, tools_list: list) -> list:
        """
        Select relevant tools from smolagents tools list.
        
        Args:
            query: User's query
            tools_list: List of smolagents Tool objects
            llm: Optional LLM instance

        Returns:
            List of selected Tool objects
        """
        if not tools_list:
            return {}
            
        # Use existing prompt_based_retrieval logic
        resources = {"tools": tools_list}
        selected = self.prompt_based_retrieval(query, resources, self.retrieval_topic, self.retrieval_guidelines)
        
        # Return the actual tool objects
        list_selected_tools = selected.get("tools", [])
        selected_tools = {t[0]: t[1] for t in list_selected_tools}
        return selected_tools

    def prompt_based_retrieval(self, query: str, resources: dict, topic='bio medial', guidelines="") -> dict:
        """Use a prompt-based approach to retrieve the most relevant resources for a query.

        Args:
            query: The user's query
            resources: A dictionary with keys 'tools', 'data_lake', and 'libraries',
                      each containing a list of available resources
            topic: Topic that supported by agents of toolbrain
            guideline: Important or specific guideline related to a specific topic

        Returns:
            A dictionary with the same keys, but containing only the most relevant resources

        """
        # Create a prompt for the LLM to select relevant resources
        tools_object = [t[1] if len(t) > 1 else t[0] for t in resources.get("tools", [])]
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
            {guidelines} 
        """.format(topic=topic, 
        query=query, 
        tools=self._format_resources_for_prompt(tools_object),
        guidelines=guidelines or "1. Focus on relevance to the query.\n2. Be comprehensive in the selection.\n3. Avoid including irrelevant items.")

        # Use the provided LLM or create a new one
        if self.llm_instance is None:
            # print(f"Using default LLM OPENAI gpt-4o-mini: {self.llm_model}")
            llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).chat.completions.create
            response = llm(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            response_content = str(response)
        else:
            # For other LLM interfaces
            # print(f"Using provided LLM: {self.llm_instance}")
            response_content = str(self.llm_instance(model=self.llm_model, messages=[{"role": "user", "content": prompt}]))  

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