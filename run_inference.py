from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:8000/v1',
    api_key="EMPTY",
)

import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo

# Simplified timezone data
TIMEZONE_DATA = {
    "tokyo": "Asia/Tokyo",
    "san francisco": "America/Los_Angeles",
    "paris": "Europe/Paris"
}

def get_current_time(location):
    """Get the current time for a given location"""
    print(f"get_current_time called with location: {location}")  
    location_lower = location.lower()
    
    for key, timezone in TIMEZONE_DATA.items():
        if key in location_lower:
            print(f"Timezone found for {key}")  
            current_time = datetime.now(ZoneInfo(timezone)).strftime("%I:%M %p")
            return json.dumps({
                "location": location,
                "current_time": current_time
            })
    
    print(f"No timezone data found for {location_lower}")  
    return json.dumps({"location": location, "current_time": "unknown"})

def run_conversation():
    # Initial user message
    messages = [{"role": "user", "content": "What's the current time in San Francisco"}] # Single function call
    #messages = [{"role": "user", "content": "What's the current time in San Francisco, Tokyo, and Paris?"}] # Parallel function call with a single tool/function defined

    # Define the function for the model
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. San Francisco",
                        },
                    },
                    "required": ["location"],
                },
            }
        }
    ]

    # First API call: Ask the model to use the function
    response = client.chat.completions.create(
        model='/workspace/Qwen2.5-7B-Instruct-bnb-4bit',
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    # Process the model's response
    response_message = response.choices[0].message
    messages.append(response_message)

    # Handle function calls
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "get_current_time":
                function_args = json.loads(tool_call.function.arguments)
                print(f"Function arguments: {function_args}")  
                time_response = get_current_time(
                    location=function_args.get("location")
                )
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": "get_current_time",
                    "content": time_response,
                })
    else:
        print("No tool calls were made by the model.")  

    # Second API call: Get the final response from the model
    final_response = client.chat.completions.create(
        model='/workspace/Qwen2.5-7B-Instruct-bnb-4bit',
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    messages.append(final_response.choices[0].message)

    for message in messages:
        print(dict(message))

    with open("conversations.json", "w") as f:
        message_dict = [message.to_dict() if not isinstance(message, dict) else message for message in messages]
        json.dump(message_dict, f)


if __name__ == "__main__":
    run_conversation()