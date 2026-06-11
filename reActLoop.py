from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

from dotenv import load_dotenv
import os

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "gemma4:latest"
OLLAMA_ENDPOINT = os.environ["OLLAMA_ENDPOINT"]

@tool
def get_price_product(product: str) -> float:
    '''A tool to get the price of a product.'''
    products = {"laptop": 1000, "phone": 500, "tablet": 300}
    return products.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    '''A tool to apply a discount and give final price.'''
    discounts = {"gold": 0.2, "silver": 0.1, "bronze": 0.05}
    discount = discounts.get(discount_tier, 0)
    return price * (1 - discount)

@traceable(name="React Loop Agent")
def run_agent(question: str):
    print(f"Question: {question}")
    print("=" * 60)
    tools = [get_price_product, apply_discount]
    tool_dict = {tool.name: tool for tool in tools}
    print(tool_dict)
    llm = init_chat_model(model = f"ollama:{MODEL}", base_url=OLLAMA_ENDPOINT)
    llm_with_tools = llm.bind_tools(tools)

    messages = [SystemMessage(content='''You are a helpful shopping assistant that can answer questions product price using the available tools.
                              Use must first call get_price_product tool to get the base price of the product.
                              Use apply_discount tool to apply discount and get the final price.
                              Always use the tools to get the price of the product and apply discount. Never guess the price.'''),
                HumanMessage(content=question)]
    
    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"Iteration {iteration}:")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        if len(tool_calls) == 0:
            print("AI Response:", ai_message.content)
            print("No tool calls, stopping.")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        observation = tool_dict[tool_name].invoke(tool_args)

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_name=tool_name, tool_call_id=tool_id))


if __name__ == "__main__":
    run_agent("What is the price of a laptop with gold discount?")