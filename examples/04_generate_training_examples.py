import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from smolagents import tool
from toolbrain import Brain, create_agent


@tool
def calculate_compound_interest(principal: float, rate: float, times_compounded: int, years: float) -> float:
    """
    Calculate compound interest.

    Args:
        principal (float): The principal amount.
        rate (float): Annual interest rate (as a decimal, e.g., 0.05 for 5%).
        times_compounded (int): Number of times interest is compounded per year.
        years (float): Number of years.

    Returns:
        float: Final amount after compounding.
    """
    return principal * (1 + rate / times_compounded) ** (times_compounded * years)


@tool
def calculate_loan_payment(principal: float, annual_rate: float, years: int) -> float:
    """
    Calculate monthly loan payment using the annuity formula.

    Args:
        principal (float): Loan amount.
        annual_rate (float): Annual interest rate (decimal, e.g., 0.06).
        years (int): Loan term in years.

    Returns:
        float: Monthly payment amount.
    """
    r = annual_rate / 12
    n = years * 12
    return principal * (r * (1 + r)**n) / ((1 + r)**n - 1)


@tool
def calculate_roi(gain: float, cost: float) -> float:
    """
    Calculate return on investment (ROI).

    Args:
        gain (float): Total gain from investment.
        cost (float): Total cost of investment.

    Returns:
        float: ROI as a decimal.
    """
    return (gain - cost) / cost


@tool
def calculate_cagr(begin_value: float, end_value: float, years: float) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR).

    Args:
        begin_value (float): Initial value.
        end_value (float): Final value.
        years (float): Number of years.

    Returns:
        float: CAGR as a decimal.
    """
    return (end_value / begin_value) ** (1 / years) - 1


@tool
def calculate_npv(rate: float, cash_flows: list[float]) -> float:
    """
    Calculate Net Present Value (NPV).

    Args:
        rate (float): Discount rate (as a decimal).
        cash_flows (list[float]): List of cash flows per period (negative for outflows, positive for inflows).

    Returns:
        float: Net present value.
    """
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows))


def main() -> None:
    description = (
        "Generate task to learn to use simple finance tools such as calculating interest, converting currencies, or computing profit. "
        "The prompts should include varied numeric inputs, realistic edge cases (zero values, high rates), and require validation of outputs."
    )
    agent = create_agent(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        tools=[calculate_compound_interest, calculate_loan_payment, calculate_cagr, calculate_npv],
    )
    brain = Brain(agent=agent)
    generated_examples = brain.generate_training_examples(
        task_description=description,
        num_examples=1,
        min_tool_calls=2,
        max_words=80,
        external_model=None,
        external_tools=None,
        guidance_example=None,
        self_rank=True
    )

    for example in generated_examples:
        print(example)
        print("")

    # import json
    # with open("generated_training_examples_ranked.json", "w") as f:
    #     json.dump(generated_examples, f, indent=4)

    # generated_examples = ["How much interest will I earn on a $10,000 investment with a 7% annual interest rate over 5 years?"]
    # training_dataset = [{"query": example} for example in generated_examples]
    # brain.train(training_dataset)

    # import json
    # generated_examples = None
    # with open("generated_training_examples.json", "r") as f:
    #     generated_examples = json.load(f)
    # print(generated_examples)

if __name__ == "__main__":
    main()