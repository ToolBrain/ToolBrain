"""
ToolBrain Training Example

This script demonstrates the new, ultra-simplified ToolBrain API:
1. Create agent with create_agent() or manually
2. Create brain with Brain() constructor (all parameters as keywords)
3. Train with explicit, self-documenting parameters

"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from smolagents import tool
from toolbrain import create_agent, Brain
from toolbrain.rewards import reward_exact_match

# --- 1. Define Tools and Reward Function (User-defined) ---
@tool
def add(a: int, b: int) -> int:
    """
    Add two integers.

    Args:
        a (int): First addend.
        b (int): Second addend.

    Returns:
        int: Sum of a and b.
    """
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two integers.

    Args:
        a (int): First factor.
        b (int): Second factor.

    Returns:
        int: Product of a and b.
    """
    return a * b

# --- 2. Prepare Training Data ---
training_dataset = [
    {
        "query": "Use the add tool to calculate 5 + 7",
        "gold_answer": "12" 
    },
    {
        "query": "What is 8 multiplied by 6?",
        "gold_answer": "48"
    },
    # Add more examples here
]

def main():
    print("🧠 ToolBrain Training Example")
    print("=" * 60)
   
    # 1. Create agent 
    agent = create_agent(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        tools=[add, multiply]  
        # use_unsloth=True (optional, default False)
    )
    
    print("✅ Agent created.")

    # 2. Create Brain 

    # === TYPICAL USAGE: Only 2 key parameters ===
    brain = Brain(
        agent,                          # Agent instance  
        algorithm="GRPO"                # Algorithm choice
        # learning_rate=3e-5 (default)
        # epsilon=0.2 (default)  
        # num_group_members=10 (default)
        # batch_size=1 (default)
        # max_grad_norm=1.0 (default)
        # use_bitsandbytes=False (default)
        # reward_func=None -> exact_match (default)
        # enable_tool_retrieval=False (default)
    )

    # User only needs to specify what they want to change

    # === DEMO: Override just 1-2 key parameters ===
    # brain_custom = Brain(
    #     agent,
    #     algorithm="GRPO",
    #     learning_rate=1e-5,             # Override learning rate
    #     num_group_members=2             # And group size for faster
    # )

    # 3. Demo other algorithms

    # DPO example - only override algorithm-specific parameters
    # dpo_brain = Brain(
    #     agent,
    #     algorithm="DPO",                # Different algorithm
    #     beta=0.04                       # Only override DPO-specific parameter
    #     # All others use defaults: learning_rate=3e-5, num_group_members=10, etc.
    # )
    
    # Supervised example
    # supervised_brain = Brain(
    #     agent,
    #     algorithm="Supervised",         # Supervised learning
    #     learning_rate=5e-5
    # )

    # === DEMO: Tool Retrieval Feature ===
    # retrieval_brain = Brain(
    #     agent,
    #     algorithm="GRPO",
    #     learning_rate=1e-5,
    #     num_group_members=2,
    #     enable_tool_retrieval=True,      # Enable intelligent tool filtering
    #     retrieval_topic="mathematics",   # Domain for tool selection
    #     retrieval_guidelines="Select only tools needed for mathematical calculations"
    # )

    # 4. Train with the GRPO brain (using brain_custom for faster)
    brain.train(training_dataset, num_iterations=1)
    
    # 5. Get trained agent
    print("\n🎉 Training completed!")
    trained_agent = brain.get_agent()
    print("✅ Trained agent is ready to use!")

    # === DEMO: Save/Load Functionality ===
    print("\n💾 Save/Load Demo")
    print("-" * 30)
    
    # Save the trained model
    model_save_path = "./saved_math_model"
    print(f"Saving model to: {model_save_path}")
    brain.save(model_save_path)
    
    # Load the saved model back
    print(f"Loading model from: {model_save_path}")
    loaded_agent = Brain.load_agent(
        model_dir=model_save_path,
        base_model_id="Qwen/Qwen2.5-0.5B-Instruct",  # Same as original
        tools=[add, multiply]  # Same tools as original
    )
    
    print("✅ Model saved and loaded successfully!")
    
    # === DEMO: Continue Training ===
    print("\n🔄 Continue Training Demo")  
    print("-" * 30)
    
    # Load model and continue training
    brain_continued = Brain.load_and_continue_training(
        model_dir=model_save_path,
        base_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        tools=[add, multiply],
        algorithm="GRPO",
        learning_rate=1e-6  # Lower learning rate for fine-tuning
    )
    
    # Additional training data
    more_training_data = [
        {
            "query": "Calculate 15 plus 25", 
            "gold_answer": "40"
        },
        {
            "query": "What's 9 times 7?",
            "gold_answer": "63" 
        }
    ]
    
    print("Training more steps...")
    brain_continued.train(more_training_data, num_iterations=1)
    print("✅ Continued training completed!")


def advanced_agent_creation_examples():
    """
    ADVANCED: Complete User Control Over Agent Creation
    
    This section demonstrates how users have FULL CONTROL over agent creation.
    ToolBrain's philosophy: Agent creation is entirely up to the user.
    Brain only handles training strategies.
    """
    
    # === OPTION 1: Factory Function (Convenience) ===
    
    # Simple factory usage
    simple_agent = create_agent(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        tools=[add, multiply],
        max_steps=10,
        use_unsloth=True
    )
    
    # Advanced factory usage with custom parameters
    advanced_factory_agent = create_agent(
        model_id="Qwen/Qwen2.5-0.5B-Instruct", 
        tools=[add, multiply],
        max_steps=20,              # Custom agent behavior
        use_unsloth=True,
        max_seq_length=20000,      # Custom model config
        max_new_tokens=1024,       # Custom model config
    )
    
    # === OPTION 2: Manual Creation ===
    
    # User creates model manually with full control
    from toolbrain.models import UnslothModel
    from smolagents import CodeAgent, TransformersModel
    
    # Option 2a: Custom UnslothModel
    custom_model = UnslothModel(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        max_seq_length=25000,      # Custom length
        max_new_tokens=2048,       # Custom generation
        # Any other model parameters
    )
    
    # User creates agent manually with full control
    manual_agent = CodeAgent(
        model=custom_model,
        tools=[add, multiply],
        max_steps=30,              # Custom agent steps
    )
    
    # Option 2b: Standard TransformersModel for comparison
    standard_model = TransformersModel(
        model_id="Qwen/Qwen2.5-0.5B-Instruct", 
        max_new_tokens=512
    )
    
    standard_agent = CodeAgent(
        model=standard_model,
        tools=[add, multiply],
        max_steps=15
    )
    
    # === OPTION 3: External Agent Libraries ===
    
    # Note: These are conceptual examples - actual usage depends on library availability
    
    # Example with AutoGen (if available)
    print("""
    from autogen import ConversableAgent
    from toolbrain.models import UnslothModel
    
    model = UnslothModel("Qwen/Qwen2.5-0.5B-Instruct")
    autogen_agent = ConversableAgent(
        name="math_assistant",
        llm_config={"model": model},
        tools=[add, multiply],
        system_message="You are a specialized math assistant."
    )
    brain = Brain(autogen_agent, algorithm="GRPO")  # Brain accepts any agent!
    """)
    
    # Example with CrewAI (if available)  
    print("""
    from crewai import Agent
    from toolbrain.models import UnslothModel
    
    model = UnslothModel("Qwen/Qwen2.5-0.5B-Instruct")
    crew_agent = Agent(
        role="mathematician",
        goal="Solve mathematical problems accurately", 
        llm=model,
        tools=[add, multiply],
        backstory="Expert in arithmetic operations"
    )
    brain = Brain(crew_agent, algorithm="GRPO")  # Works with any agent library!
    """)
    
    # === OPTION 4: Custom Agent Classes ===
    
    # User defines completely custom agent
    class MyCustomMathAgent:
        """
        User-defined custom agent with specialized behavior.
        This demonstrates complete freedom in agent design.
        """
        def __init__(self, model, tools, reasoning_style="step_by_step"):
            self.model = model
            self.tools = tools
            self.reasoning_style = reasoning_style
            self.memory = []
            self.custom_behavior = "mathematical_reasoning"
        
        def run(self, query):
            """Custom execution logic defined by user"""
            self.memory.append(f"Processing: {query}")
            
            if self.reasoning_style == "step_by_step":
                return self._step_by_step_reasoning(query)
            else:
                return self._direct_reasoning(query)
        
        def _step_by_step_reasoning(self, query):
            """User's custom reasoning approach"""
            steps = [
                "1. Analyze the mathematical problem",
                "2. Identify required operations", 
                "3. Execute calculations step by step",
                "4. Verify the result"
            ]
            # Custom logic here
            return f"Step-by-step solution for: {query}"
        
        def _direct_reasoning(self, query):
            """Alternative reasoning approach"""
            return f"Direct solution for: {query}"
        
        def get_memory(self):
            """Custom memory management"""
            return self.memory
    
    # User creates custom agent instance
    custom_model = UnslothModel("Qwen/Qwen2.5-0.5B-Instruct")
    custom_agent = MyCustomMathAgent(
        model=custom_model,
        tools=[add, multiply], 
        reasoning_style="step_by_step"
    )
    
    # === OPTION 5: Enhanced Existing Agents ===
    
    # User extends existing agent with additional capabilities
    class EnhancedCodeAgent(CodeAgent):
        """
        User extends smolagents CodeAgent with custom enhancements.
        """
        def __init__(self, model, tools, confidence_threshold=0.8):
            super().__init__(model, tools)
            self.confidence_threshold = confidence_threshold
            self.execution_history = []
        
        def run(self, query):
            """Enhanced run method with custom logic"""
            result = super().run(query)  # Use base functionality
            
            # Add custom post-processing
            self.execution_history.append({
                "query": query,
                "result": result,
                "confidence": self._calculate_confidence(result)
            })
            
            return result
        
        def _calculate_confidence(self, result):
            """User's custom confidence calculation"""
            # Custom logic to assess result confidence
            return 0.9  # Placeholder
        
        def get_execution_history(self):
            """Custom method for tracking execution"""
            return self.execution_history
    
    # User creates enhanced agent
    enhanced_model = UnslothModel("Qwen/Qwen2.5-0.5B-Instruct")
    enhanced_agent = EnhancedCodeAgent(
        model=enhanced_model,
        tools=[add, multiply],
        confidence_threshold=0.85
    )


if __name__ == "__main__":
    main()
    
    # advanced_agent_creation_examples()