from llama3asyncfunc import startWorker, prompt

# model_id = 'Groq/Llama-3-Groq-8B-Tool-Use'
model_id = "meta-llama/Llama-3.2-3B-Instruct"
startWorker(model_id)

from agent import startAgent


# initialPrompt = [
#         {
#             "role": "system",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": 
# """
# You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
# <tool_call>
# {"name": <function-name>,"arguments": <args-dict>}
# </tool_call>

# Here are the available tools:
# <tools> {
#     "name": "get_current_weather",
#     "description": "Get the current weather in a given location",
#     "parameters": {
#         "properties": {
#             "location": {
#                 "description": "The city and state, e.g. San Francisco, CA",
#                 "type": "string"
#             },
#             "unit": {
#                 "enum": [
#                     "celsius",
#                     "fahrenheit"
#                 ],
#                 "type": "string"
#             }
#         },
#         "required": [
#             "location"
#         ],
#         "type": "object"
#     }
# } </tools>"""
#                 },
#             ],
#         },
# ]

# startAgent(prompt,'function', initialPrompt=initialPrompt)
startAgent(prompt,'function',initialPrompt=[{"role": "system", "content": [{"type":"text","text":"You are a function calling AI model."}]}])    