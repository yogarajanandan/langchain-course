from langsmith import traceable
from ollama import Client as OllamaClient
from dotenv import load_dotenv
import os

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "gemma4:latest"
OLLAMA_ENDPOINT = os.environ["OLLAMA_ENDPOINT"]
ollama_client = OllamaClient(host=OLLAMA_ENDPOINT)

@traceable(run_type="tool")
def get_price_product(product: str) -> float:
    '''A tool to get the price of a product.'''
    products = {"laptop": 1000, "phone": 500, "tablet": 300}
    return products.get(product, 0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    '''A tool to apply a discount and give final price.'''
    discounts = {"gold": 0.2, "silver": 0.1, "bronze": 0.05}
    discount = discounts.get(discount_tier, 0)
    return price * (1 - discount)

tools_for_llm = [ 
    {
        "type": "function",
        "function": {
                "name": "get_price_product",
                "description": "A tool to get the price of a product.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product": {
                            "type": "string",
                            "description": "The name of the product to get the price for."
                        }
                    },
                    "required": ["product"]
                }
                }
    
    },
    {
        "type": "function",
        "function": {
                "name": "apply_discount",
                "description": "A tool to apply a discount and give final price.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "price": {
                            "type": "number",
                            "description": "The base price of the product."
                        },
                        "discount_tier": {
                            "type": "string",
                            "description": "The discount tier to apply (gold, silver, bronze)."
                        }
                    },
                    "required": ["price", "discount_tier"]
                }
                }
    
    }
]

@traceable(run_type="llm")
def ollama_chat(messages, tools):
    response = ollama_client.chat(model=MODEL, messages=messages, tools=tools)
    return response

@traceable(name="Raw Function Call Agent")
def run_agent(question: str):
    print(f"Question: {question}")
    print("=" * 60)

    tools_dict = {"get_price_product": get_price_product, "apply_discount": apply_discount}
    
    messages = [{"role": "system", "content": '''You are a helpful shopping assistant that can answer questions product price using the available tools.
                              Use must first call get_price_product tool to get the base price of the product.
                              Use apply_discount tool to apply discount and get the final price.
                              Always use the tools to get the price of the product and apply discount. Never guess the price.'''},
                {"role": "user", "content": question}]
    

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"Iteration {iteration}:")

        ollama_response = ollama_chat(messages=messages, tools=tools_for_llm)
        ai_message = ollama_response.message

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print("AI Response:", ai_message.content)
            print("No tool calls, stopping.")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        #tool_id = tool_call["id"]

        observation = tools_dict[tool_name](**tool_args)

        messages.append(ai_message)
        messages.append({"role": "tool", "name": tool_name, "content": str(observation)})


if __name__ == "__main__":
    run_agent("What is the price of a laptop with gold discount?")