"""
Brain module - Core training orchestration for ToolBrain.

This module contains the Brain class which orchestrates the training process
by coordinating agent execution, reward calculation, and RL algorithm updates.
"""

from typing import Any, Callable, List, Protocol
import functools

from .types import Trace, TraceStep


class Agent(Protocol):
    """Protocol defining the interface for agents used with Brain."""
    def run(self, query: str) -> Any:
        """Execute a query and return the result."""
        ...


class RLAlgorithm(Protocol):
    """Protocol defining the interface for RL algorithms."""
    def train_step(self, traces: List[Trace], rewards: List[float]) -> None:
        """Perform a single training step given traces and rewards."""
        ...


class MockRLAlgorithm:
    """Mock RL algorithm for development and testing."""
    
    def __init__(self, algorithm_name: str = "MockRL") -> None:
        self.algorithm_name = algorithm_name
        self.training_steps = 0
    
    def train_step(self, traces: List[Trace], rewards: List[float]) -> None:
        """Simulate a training step by logging trace and reward information."""
        self.training_steps += 1
        
        print("=" * 60)
        print(f"🧠 {self.algorithm_name} - Training Step #{self.training_steps}")
        print("=" * 60)
        print(f"📊 Batch Size: {len(traces)} traces")
        print(f"📈 Average Reward: {sum(rewards) / len(rewards):.3f}")
        print(f"📊 Reward Range: [{min(rewards):.3f}, {max(rewards):.3f}]")
        print()
        print("📋 Training Data Summary:")
        
        for i, (trace, reward) in enumerate(zip(traces, rewards)):
            print(f"  Trace {i+1}: {len(trace)} steps, reward = {reward:.3f}")
            for j, step in enumerate(trace[:6]):  
                content_preview = step["content"][:50] + "..." if len(step["content"]) > 50 else step["content"]
                print(f"    Step {j+1} [{step['type']}]: {content_preview}")
            
            if len(trace) > 5:
                print(f"    ... and {len(trace) - 5} more steps")
            print()  # Add spacing between traces
        
        print()
        print("🔄 Mock training completed. Real RL algorithm would update model weights here.")
        print("=" * 60)


class Brain:
    """
    Core training orchestrator for ToolBrain.
    
    The Brain automatically instruments any CodeAgent to capture execution traces,
    hiding all complexity from the user. Users simply pass their normal CodeAgent
    and Brain handles the rest.
    """
    
    def __init__(
        self,
        agent: Agent,
        reward_func: Callable[[Trace, str], float],
        learning_algorithm: str = "MockRL"
    ) -> None:
        """
        Initialize the Brain with automatic agent instrumentation.
        
        Args:
            agent: Any CodeAgent instance (will be automatically instrumented)
            reward_func: Function to calculate rewards from traces
            learning_algorithm: Name of the RL algorithm to use
        """
        self.reward_func = reward_func
        self.learning_algorithm = learning_algorithm
        
        # Automatically instrument the agent for trace capture
        self.agent = self._instrument_agent(agent)
        
        # Initialize RL algorithm (currently only MockRL is implemented)
        self.rl_module: RLAlgorithm = MockRLAlgorithm(learning_algorithm)
        
        print(f"🧠 Brain initialized with {learning_algorithm} algorithm")
        print("✅ Agent automatically instrumented for tracing")
    
    def _instrument_agent(self, agent_instance: Agent) -> Agent:
        """
        Automatically instrument a CodeAgent to capture execution traces.
        
        This method uses monkey patching to wrap the agent's run() method,
        allowing it to capture traces without requiring user changes.
        
        Args:
            agent_instance: The original CodeAgent to instrument
            
        Returns:
            The same agent instance, now with tracing capability
        """
        # Store the original run method
        original_run = agent_instance.run
        
        def traced_run(query: str, **kwargs) -> Trace:
            """Wrapper that captures execution traces from the agent."""
            print(f"Agent executing: {query}")
            
            try:
                # Execute the original run method with reset=True to clear memory
                result = original_run(query, reset=True, **kwargs)
                
                # Extract trace from agent's memory
                trace = self._extract_trace_from_memory(agent_instance)
                
                # Ensure we have a final answer step
                if not any(step["type"] == "final_answer" for step in trace):
                    trace.append({
                        "type": "final_answer",
                        "content": str(result) if result is not None else "No result"
                    })
                
                return trace
                
            except Exception as e:
                # Even if execution fails, return a trace showing what happened
                trace = self._extract_trace_from_memory(agent_instance)
                trace.append({
                    "type": "final_answer", 
                    "content": f"Error: {str(e)}"
                })
                return trace
        
        # Replace the agent's run method with our traced version
        agent_instance.run = traced_run
        
        return agent_instance
    
    def _extract_trace_from_memory(self, agent_instance: Agent) -> Trace:
        """
        Extract structured trace from agent's internal memory.
        
        Args:
            agent_instance: The agent to extract traces from
            
        Returns:
            Structured trace of the agent's execution
        """
        trace: Trace = []
        
        try:
            if hasattr(agent_instance, 'memory') and hasattr(agent_instance.memory, 'steps'):
                for step in agent_instance.memory.steps:
                    # TaskStep (user query)
                    if hasattr(step, '__class__') and 'TaskStep' in str(step.__class__):
                        if hasattr(step, 'task'):
                            trace.append({
                                "type": "thought", 
                                "content": f"Processing task: {step.task}"
                            })
                    
                    # ActionStep (agent action)
                    elif hasattr(step, '__class__') and 'ActionStep' in str(step.__class__):
                        # Extract thought from model output
                        if hasattr(step, 'model_output') and step.model_output:
                            model_output = str(step.model_output)
                            if 'Thought:' in model_output:
                                thought_part = model_output.split('Thought:')[1]
                                if 'Code:' in thought_part:
                                    thought_part = thought_part.split('Code:')[0]
                                thought_part = thought_part.strip()
                                if thought_part:
                                    trace.append({"type": "thought", "content": thought_part})
                        
                        # Extract tool code
                        if hasattr(step, 'code_action') and step.code_action:
                            action_content = str(step.code_action)
                            action_content = '\n'.join(line.strip() for line in action_content.split('\n') if line.strip())
                            if action_content:  # Only add non-empty code
                                trace.append({
                                    "type": "tool_code", 
                                    "content": action_content
                                })
                        
                        # Extract tool output from action_output
                        if hasattr(step, 'action_output') and step.action_output is not None:
                            output_content = str(step.action_output)
                            if output_content.strip():
                                trace.append({
                                    "type": "tool_output", 
                                    "content": output_content
                                })
                        
                        # Extract errors as tool_output
                        if hasattr(step, 'error') and step.error:
                            error_content = str(step.error)
                            trace.append({
                                "type": "tool_output", 
                                "content": f"Error: {error_content}"
                            })
                        
                        # Check for observations (additional outputs)
                        if hasattr(step, 'observations') and step.observations:
                            # Join all observations into a single coherent output
                            all_observations = []
                            for obs in step.observations:
                                if obs and str(obs).strip():
                                    all_observations.append(str(obs).strip())
                            
                            if all_observations:
                                # Combine all observations into one coherent tool_output
                                combined_output = ''.join(all_observations)
                                if combined_output.strip():
                                    trace.append({
                                        "type": "tool_output", 
                                        "content": combined_output.strip()
                                    })
            
        except Exception as e:
            trace.append({
                "type": "thought", 
                "content": f"Memory parsing error: {str(e)}"
            })
        
        # Fallback if no steps were captured
        if not trace:
            trace = [
                {"type": "thought", "content": "Processing the query..."},
                {"type": "tool_code", "content": "Executing requested operation"},
                {"type": "tool_output", "content": "Operation completed"}
            ]
        
        return trace
    
    def train_step(self, query: str, gold_answer: str, num_group_members: int = 10) -> None:
        """
        Execute a single training step.
        
        Args:
            query: The input query for the agent
            gold_answer: The expected correct answer
            num_group_members: Number of agent runs to collect for training
        """
        print(f"\n🚀 Starting training step with query: '{query}'")
        print(f"🎯 Expected answer: '{gold_answer}'")
        print(f"👥 Collecting {num_group_members} traces...")
        
        traces: List[Trace] = []
        rewards: List[float] = []
        
        # Collect multiple traces by running the agent multiple times
        for i in range(num_group_members):
            print(f"  🔄 Running agent iteration {i+1}/{num_group_members}...")
            
            try:
                trace = self.agent.run(query)
                reward = self.reward_func(trace, gold_answer)
                
                traces.append(trace)
                rewards.append(reward)
                
                print(f"    ✅ Trace {i+1}: {len(trace)} steps, reward = {reward:.3f}")
                
            except Exception as e:
                print(f"    ❌ Error in iteration {i+1}: {e}")
                continue
        
        if not traces:
            raise RuntimeError("No successful traces collected for training")
        
        avg_reward = sum(rewards) / len(rewards)
        print(f"\n📊 Collected {len(traces)} traces with average reward {avg_reward:.3f}")
        
        # Pass traces and rewards to the RL algorithm
        self.rl_module.train_step(traces, rewards)
        
        print("\n✅ Training step completed!")
    
    def get_training_stats(self) -> dict:
        """Get current training statistics."""
        return {
            "algorithm": self.learning_algorithm,
            "training_steps": self.rl_module.training_steps
        } 