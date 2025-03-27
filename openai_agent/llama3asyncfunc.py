
import logging
import asyncio
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re

#logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

from multiprocessing import Process, Queue
import asyncio
# import transformers
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,pipeline


def control_lights(room: str, state: str):
    """
    Control the lights in a room

    Args:
        room: The room to control the lights in specified as a string of either "study" or "bedroom"
        state: The state to set the lights to specified as a string of either "on" or "off"
    """
    # return f"Turning the lights in the {room} {state}"
    return f"{state}"


chat_template="""\
{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = false %}
{%- endif %}
{%- if not date_string is defined %}
    {%- if strftime_now is defined %}
        {%- set date_string = strftime_now("%d %b %Y") %}
    {%- else %}
        {%- set date_string = "24 September 2024" %}
    {%- endif %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message + environment setup (modified for 3.2) #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if tools is not none %}
    {{- "You are an expert in composing functions. You are given a question and a set of possible functions.\n" }}
    {{- "Based on the question, you will need to make one or more function/tool calls to achieve the purpose.\n" }}
    {{- "If none of the function can be used, point it out. If the given question lacks the parameters required by the function,\n" }}
    {{- "also point it out. You should only return the function call in tools call sections.\n\n" }}
    {{- "If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\n" }}
    {{- "You SHOULD NOT include any other text in the response.\n\n" }}
    {{- "Here is a list of functions in JSON format that you can invoke.[\n" }}
    {%- for t in tools %}
        {%- set tool_json = t['function'] | tojson(indent=4) %}
        {{- "    " + tool_json | replace('\n', '\n    ') }}
        {{- ",\n" if not loop.last else "\n" }}
    {%- endfor %}
    {{- "]" }}
{%- elif builtin_tools is defined %}
    {{- "Environment: ipython\n" }}
    {{- "Tools: " + builtin_tools | reject('equalto', 'code_interpreter') | join(", ") }}
    {{- "Cutting Knowledge Date: December 2023\n" }}
    {{- "Today Date: " + date_string + "\n\n" }}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools passed in a user message (modified for consistency with 3.2) #}
{%- if tools_in_user_message and tools is not none %}
    {#- Extract the first user message to include tools if necessary #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
    {%- endif %}
    {{- "<|start_header_id|>user<|end_header_id|>\n\n" }}
    {{- "You are an expert in composing functions. You are given a question and a set of possible functions.\n" }}
    {{- "Based on the question, you will need to make one or more function/tool calls to achieve the purpose.\n" }}
    {{- "If none of the function can be used, point it out. If the given question lacks the parameters required by the function,\n" }}
    {{- "also point it out. You should only return the function call in tools call sections.\n\n" }}
    {{- "If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\n" }}
    {{- "You SHOULD NOT include any other text in the response.\n\n" }}
    {{- "Here is a list of functions in JSON format that you can invoke.[\n" }}
    {%- for t in tools %}
        {%- set tool_json = t['function'] | tojson(indent=4) %}
        {{- "    " + tool_json | replace('\n', '\n    ') }}
        {{- ",\n" if not loop.last else "\n" }}
    {%- endfor %}
    {{- "]" }}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{#- Loop through messages to handle user, assistant, and tool interactions -#}
{%- for message in messages %}
    {%- if message.role == 'user' %}
        {{- '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}
    {%- elif message.role == 'assistant' and 'tool_calls' in message %}
        {#- Handle zero-shot tool calls (modified for 3.2) #}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
        {%- for tool_call in message.tool_calls %}
            {{- '<|python_tag|>[' + tool_call.function.name + '(' }}
            {%- for arg_name, arg_val in tool_call.function.arguments.items() %}
                {{- arg_name + '=' + arg_val}}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- endif %}
            {%- endfor %}
            {{- ')]' }}
        {%- endfor %}
        {{- '<|eot_id|>' }}
    {%- elif message.role == 'assistant' %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}
    {%- elif message.role == 'tool' %}
        {#- Modified for 3.2 #}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- elif message.role == 'code' %}
        {#- Code interpreter handling (maintained from 3.1) #}
        {{- "<|python_tag|>" + message['content'] | trim + "<|eom_id|>" }}
    {%- endif %}
{%- endfor %}

{#- Add the prompt for generation if specified -#}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}\
"""

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     #model_kwargs={"torch_dtype": torch.bfloat16},
#     model_kwargs={"torch_dtype": torch.float16},
#     device_map="auto",
# )

# messages = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "Who was George Washington?"},
# ]

# messages = [
#             {
#                 "role": "system",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": "You are a pirate chatbot who always responds in pirate speak!"
#                     },                    
#                 ],
#             },
#             {
#                 "role": "user", 
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": "Who was George Washington?"
#                     },   
#                 ]
#             }
#         ]



# # outputs = pipeline(
# #     messages,
# #     max_new_tokens=256,
# # )
# print(outputs[0]["generated_text"][-1])

class ToolPipeline:
    def __init__(self, tools):
        self.tools = tools
        # self.model_id = "meta-llama/Llama-3.2-3B-Instruct"
        self.model_id = "meta-llama/Llama-3.1-8B-Instruct"
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.model_id,
        #     # load_in_8bit=True,
        #     load_in_4bit=True,
        #     device_map="auto"
        # )

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.chat_template = chat_template
        self.model = self.model.eval()

        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)             
        
    def gen(self, prompt, model, tokenizer, add_special_tokens=True):
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True)

        result = outputs[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(result, skip_special_tokens=True)

    def generate(self, prompt):

        def meta_tool_dict(func_dict, f, func_name, **kwargs):
            # args = {k: str(v) for k, v in dict(**kwargs).items()}
            func_dict[func_name] = {"name": func_name, "arguments": dict(**kwargs)}
            return f

        async def handle_coroutine(v):
            return await v if asyncio.iscoroutine(v) else v
        
        def coerce_args_to_str(tool_call):
            tool_call['arguments'] = {k: str(v) for k, v in tool_call['arguments'].items()}
            return tool_call

        prompt = prompt.copy()

        print('prompt:',   prompt)
        input = self.tokenizer.apply_chat_template(
            prompt,
            tools=self.tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        input = input.replace("object", "dict")

        print('input:', input)

        tool_call_str = self.gen(input, self.model, self.tokenizer, add_special_tokens=False)
        match = re.match(r'^(\[.*\])<\|eot_id\|>$', tool_call_str, re.DOTALL)
        if match:
            tool_call_str = match.group(1)
        # print('tool_call_str:', tool_call_str)
        functions_dict = {}
        meta_tool_name = partial(meta_tool_dict, functions_dict)
        # create as dictionary so I can use to sandbox eval
        meta_dict = {f.__name__ : partial(meta_tool_name,f,f.__name__) for f in self.tools}

        print('tool_call_str:', tool_call_str)

        try:
            invoked_functions = eval(tool_call_str, meta_dict)
            print('invoked_functions:', invoked_functions)
            # results = [await handle_coroutine(f(**functions_dict[f.__name__]['arguments'])) for f in invoked_functions]
            results = [(f(**functions_dict[f.__name__]['arguments'])) for f in invoked_functions]
            tool_call = [{"type":"function", "function": coerce_args_to_str(functions_dict[f.__name__])} for f in invoked_functions]
        except Exception as e:
            print('Exception:', e)
            results = []
            tool_call = []

        print('tool_call:', tool_call)

        prompt.append({"role": "assistant", "tool_calls": tool_call})
        # prompt.append({"role": "tool", "content": json.dumps(results[0])})
        prompt.append({"role": "tool", "content": results[0]})

        input = self.tokenizer.apply_chat_template(
            prompt,
            tools=self.tools,
            tokenize=False,
            add_generation_= True,
        )
        input = input.replace("object", "dict")
        print('input:', input)

        output = self.gen(input, self.model, self.tokenizer, add_special_tokens=False)
        # prompt.append({"role": "assistant", "content": output})
        # print('prompt:', prompt)
        # print('prompt:', prompt[-1]['content'])
        # print('output:', output)        
        return output


class Worker:
    def __init__(self,model_id=None):
        # if model_id is None:
        #     model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        # else:
        #     self.model_id = model_id
        self.lock = asyncio.Lock()
        self.inQ = Queue()
        self.outQ = Queue()
        self.p = Process(
            target=self.loop,
            args=(self.inQ, self.outQ, model_id))
        self.p.start()

    @staticmethod
    def loop(inQ: Queue, outQ: Queue, model_id: str):
        try:
            # asr = pipeline("automatic-speech-recognition",model="openai/whisper-large-v3",device=0)
            # asr = pipeline("automatic-speech-recognition",model="openai/whisper-tiny.en",device=0)
            # asr = pipeline("automatic-speech-recognition",model="openai/whisper-tiny",device='cpu')
            

            # model_name = "meta-llama/Llama-3.1-8B"
            # model = AutoModelForCausalLM.from_pretrained(
            #     model_id,
            #     # load_in_8bit=True,
            #     load_in_4bit=True,
            #     device_map="auto"
            # )

            # tokenizer = AutoTokenizer.from_pretrained(model_id)            
            # pl = pipeline(
            #     "text-generation",
            #     model=model_id,
            #     #model_kwargs={"torch_dtype": torch.bfloat16},
            #     model_kwargs={"torch_dtype": torch.float16},
            #     device_map="auto",
            # )

            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # print(f"Model loaded on device: {device}")


            # def generate(prompt, model, tokenizer, add_special_tokens=True):
            #     print('Prompt:',prompt)
            #     prompt = prompt.copy()
            #     prompt = tokenizer.apply_chat_template(
            #         prompt,
            #         # tools=self.tools,
            #         tokenize=False,
            #         add_generation_prompt=True,
            #     )
            #     inputs = tokenizer(
            #         prompt,
            #         return_tensors="pt",
            #         add_special_tokens=add_special_tokens,
            #     ).to(model.device)

                # with torch.no_grad():
                #     outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True)

                # result = outputs[0][inputs["input_ids"].shape[-1]:]
                # return tokenizer.decode(result, skip_special_tokens=True)


            # pl = pipeline(
            #     task="text-generation",
            #     model=model,
            #     tokenizer=tokenizer
            # )
            # 
            pl = ToolPipeline([control_lights])            
            
            print('Starting Worker',pipeline)

            while True:
                command, args = inQ.get()
                print('Command:',command)
                if command == 'stop':
                    break
                elif command == 'prompt':
                    #outQ.put((asr(args[0])))
                    # print('args:',args)
                    # o = pl(args,max_new_tokens=256)
                    o = pl.generate(args[0])
                    print('Output:',o)
                    # print('Output:',o)
                    outQ.put(o)

        except KeyboardInterrupt:
            pass

    def stop(self):
        self.send('stop')
        self.p.join()
        self.inQ.close()
        self.outQ.close()

    async def send(self,*task):
        async with self.lock:
            self.inQ.put(task)
            return await asyncio.get_running_loop().run_in_executor(None, self.outQ.get)

def startWorker(model_id=None):
    global w
    w = Worker(model_id)

async def prompt(messages):
    # print('Prompting')
    print('Prompting1:',messages)
    messages = [{"role":r['role'],"content":r['content'][0]['text']} for r in messages]
    print('Prompting2:',messages)
    r = (await w.send('prompt',(messages,)))#[0][0]['generated_text'][-1]['content']
    print('llama response:', r)
    print('Type of r:', type(r))  # Debug statement to check the type of r
    # check if r is a string
    if isinstance(r, str):
        print('string')
        return r
    # else check r is a list
    elif isinstance(r, list):
        print('list')
        return r[0]['text']
    else:
        print('object')
        return r['text']

# startWorker()

# pretty print json
# import json
# print(json.dumps(asyncio.run(prompt(messages))[0][0]['generated_text'][-1]['content'], indent=2))