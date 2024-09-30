# Copyright 2024 John Robinson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
#logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

import datetime
import json

from openai import AsyncOpenAI
# from auth_openai import api_key
from config import config
openai_api_key = config.get('openai_api_key')

client = AsyncOpenAI(api_key=openai_api_key)

#dirty
local_dt = datetime.datetime.utcnow()
date_time_string = local_dt.strftime("%A, %B %d, %Y %H:%M:%S %Z UTC")

#dirty
prompt_messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": f"You are a helpful AI assistant.  The current date and time is {date_time_string}.  When reporting the time or date, speak succinctly.  When telling a joke, put the whole joke on the first line."
            },
        ],
    },
]

messageListener = None

def setMessageListener(l):
    global messageListener
    messageListener = l

# user,assistant,system
async def appendMessage(m):
    prompt_messages.append(m)
    if messageListener:
        await messageListener(m if isinstance(m, dict) else m.__dict__)

def getMessages():
    return prompt_messages        

async def prompt(t):
    await appendMessage({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": t
                },
            ],
        })
    

    # Example dummy function hard coded to return the same weather
    # In production, this could be your backend API or an external API
    def get_current_weather(location, unit="fahrenheit"):
        """Get the current weather in a given location"""
        if "tokyo" in location.lower():
            return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
        elif "san francisco" in location.lower():
            return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
        elif "paris" in location.lower():
            return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
        else:
            return json.dumps({"location": location, "temperature": "unknown"})
        
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]        

    response = await client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=prompt_messages,
        tools=tools,
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_current_weather": get_current_weather,
        }  # only one function in this example, but you can have multiple
        #messages.append(response_message)  # extend conversation with assistant's reply
        # print('xx:', response_message)
        # print('yy:', response_message.__dict__)
        #await appendMessage(response_message)
        prompt_messages.append(response_message)
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                location=function_args.get("location"),
                unit=function_args.get("unit"),
            )
            await appendMessage(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }                
            )
        response = await client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=prompt_messages,
        )  # get a new response from the model where it can see the function response
        print('second call response:', response)
        response_message = response.choices[0].message
    await appendMessage({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": response_message.content
            },
        ],
    })
