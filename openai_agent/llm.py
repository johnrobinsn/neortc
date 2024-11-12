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

import asyncio
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

from datetime import datetime
import json
import os
import copy

from openai import AsyncOpenAI

from llama3async import startWorker, prompt as llama3prompt


startWorker()

# from auth_openai import api_key
from aconfig import config
import uuid

from abc import ABC, abstractmethod

openai_api_key = config.get('openai_api_key')

class ILLM(ABC):
    @abstractmethod
    async def setName(self,n):
        pass

    @abstractmethod
    def addListener(self,l):
        pass

    @abstractmethod
    def delListener(self,l):
        pass

    @abstractmethod
    def addMetaDatalistener(self,l):
        pass

    @abstractmethod
    def delMetaDataListener(self,l):
        pass

    @abstractmethod
    async def appendMessage(self,m):
        pass

    @abstractmethod
    def getMessages(self):
        pass

    @abstractmethod
    async def prompt(self,t):
        pass

    @abstractmethod
    def getMetaData(self):
        pass

    async def notifyMetaDataChanged(self):
        pass

    @abstractmethod
    async def summarize(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self,path):
        pass

class LLM(ILLM):
    client = AsyncOpenAI(api_key=openai_api_key)
    nameIndex = 0
    def __init__(self):
        self.created = datetime.utcnow().isoformat() + "Z"
        self.id = str(uuid.uuid4())
        self.name = ''
        self.summary = ''
        self.persisted = False
        self.dead = False
        LLM.nameIndex += 1

        self.local_dt = datetime.utcnow() #now(datetime.timezone.utc)
        self.date_time_string = self.local_dt.strftime("%A, %B %d, %Y %H:%M:%S %Z UTC")
        self.prompt_messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"You are a helpful AI assistant.  The current date and time is {self.date_time_string}.  When reporting the time or date, speak succinctly.  When telling a joke, put the whole joke on the first line."
                    },
                ],
            },
        ]
        self.listeners = []
        self.metaDataListeners = []
        # self.save()

    def save(self):
        j = {
            'id':self.id,
            'name':self.name,
            'summary':self.summary,
            'created':self.created,
            'log':self.prompt_messages,
        }
        os.makedirs('openai_chats', exist_ok=True)
        with open(f'openai_chats/{self.id}.json', 'w') as file:
            file.write(json.dumps(j))
        self.persisted = True

    def load(self,path):
        with open(path,'r') as file:
            j = json.load(file)
            self.id = j.get('id','')
            self.name = j.get('name','')
            self.summary = j.get('summary','')
            self.created = j.get('created','')
            self.prompt_messages = j.get('log',[])
        self.persisted = True

    def getMetaData(self):
        display = self.name
        if not display: display = self.summary
        if not display: display = 'Untitled'
        return {'id':self.id,'display':display,'summary':self.summary,'name':self.name,'created':self.created}
    
    async def notifyMetaDataChanged(self):
        for l in self.metaDataListeners:
            await l(self.getMetaData())
        # self.save()       

    async def setName(self,n):
        self.name = n
        await self.notifyMetaDataChanged()

    def addListener(self,l):
        if l:
            self.listeners.append(l)
    
    def delListener(self, l):
        if l in self.listeners:
            self.listeners.remove(l)
        if len(self.listeners) == 0 and not self.persisted:
            self.dead = True
        #asyncio.run(self.notifyMetaDataChanged())
        asyncio.get_event_loop().create_task(self.notifyMetaDataChanged())

    def addMetaDatalistener(self,l):
        if l:
            self.metaDataListeners.append(l)
            # self.notifyMetaDataChanged()

    def delMetaDataListener(self,l):
        if l in self.metaDataListeners:
            self.metaDataListeners.remove(l)

    # user,assistant,system
    async def appendMessage(self,m):
        self.prompt_messages.append(m)
        for l in self.listeners:
            await l(m if isinstance(m, dict) else m.__dict__)
        
        # ask model to summarize conversation
        summary = await self.summarize()
        print("Summary:", summary)
        if self.summary != summary:
            self.summary = summary
            await self.notifyMetaDataChanged()
        self.save()

    def getMessages(self):
        return self.prompt_messages

    async def summarize(self):
        log = copy.deepcopy(self.prompt_messages)
        log.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": 'Can you summarize the conversation so far in four words or less?'
                },
            ],
        })
        # response = await LLM.client.chat.completions.create(
        #     model="gpt-4-1106-preview",
        #     messages=log,
        # )
        # response_message = response.choices[0].message
        # return response_message.content  
        return await llama3prompt(log)             

    async def prompt(self,t):
        await self.appendMessage({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": t
                    },
                ],
            })

        start = datetime.now()

        log.info('Time for LLM Response: %d', (datetime.now()-start).total_seconds())

        # response_message = response.choices[0].message
        # await self.appendMessage({
        #     "role": "assistant",
        #     "content": [
        #         {
        #             "type": "text",
        #             "text": response_message.content
        #         },
        #     ],
        # })

        #llama
        response_message = await llama3prompt(self.prompt_messages)
        # response_message = response[0][0]['generated_text'][-1]['content']
        #openai
        # response = await LLM.client.chat.completions.create(
        #     model="gpt-4-1106-preview",
        #     messages=log,
        # )
        # response_message = response.choices[0].message.content

        await self.appendMessage({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": response_message
                },
            ],
        })        


class OpenAI_LLM(LLM):
    def __init__(self):
        super().__init__()
        self.client = AsyncOpenAI(api_key=openai_api_key)
        # self.name = 'OpenAI'
        # self.prompt_messages = [
        #     {
        #         "role": "system",
        #         "content": [
        #             {
        #                 "type": "text",
        #                 "text": f"Welcome to OpenAI's chat service.  The current date and time is {self.date_time_string}.  When reporting the time or date, speak succinctly.  When telling a joke, put the whole joke on the first line."
        #             },
        #         ],
        #     },
        # ]

    # async def prompt(self,t):
    #     await self.appendMessage({
    #             "role": "user",
    #             "content": [
    #                 {
    #                     "type": "text",
    #                     "text": t
    #                 },
    #             ],
    #         })

    #     # Example dummy function hard coded to return the same weather
    #     # In production, this could be your backend API or an external API
    #     def get_current_weather(location, unit="fahrenheit"):
    #         """Get the current weather in a given location"""
    #         if "tokyo" in location.lower():
    #             return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    #         elif "san francisco" in location.lower():
    #             return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    #         elif "paris" in location.lower():
    #             return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    #         else:
    #             return json.dumps({"location": location, "temperature": "unknown"})
            
    #     tools = [
    #         {
    #             "type": "function",
    #             "function": {
    #                 "name": "get_current_weather",
    #                 "description": "Get the current weather in a given location",
    #                 "parameters": {
    #                     "type": "object",
    #                     "properties": {
    #                         "location": {
    #                             "type": "string",
    #                             "description": "The city and state, e.g. San Francisco, CA",
    #                         },
    #                         "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    #                     },
    #                     "required": ["location"],
    #                 },
    #             },
    #         }
    #     ]        

    #     start = datetime.now()
    #     response = await LLM.client.chat.completions.create(
    #         model="gpt-4-1106-preview",
    #         messages=self.prompt_messages,
    #         tools=tools,
    #     )
    #     log.info('Time for LLM Response: %d', (datetime.now()-start).total_seconds())

    #     response_message = response.choices[0].message
    #     tool_calls = response_message.tool_calls
    #     if tool_calls:
    #         # Step 3: call the function
    #         # Note: the JSON response may not always be valid; be sure to handle errors
    #         available_functions = {
    #             "get_current_weather": get_current_weather,
    #         }  # only one function in this example, but you can have multiple
    #         #messages.append(response_message)  # extend conversation with assistant's reply
    #         # print('xx:', response_message)
    #         # print('yy:', response_message.__dict__)
    #         #await appendMessage(response_message)
    #         self.prompt_messages.append(response_message)
    #         # Step 4: send the info for each function call and function response to the model
    #         for tool_call in tool_calls:
    #             print("Calling tools")
    #             function_name = tool_call.function.name
    #             function_to_call = available_functions[function_name]
    #             function_args = json.loads(tool_call.function.arguments)
    #             function_response = function_to_call(
    #                 location=function_args.get("location"),
    #                 unit=function_args.get("unit"),
    #             )
    #             # if function_response:
    #             await self.appendMessage(
    #                 {
    #                     "tool_call_id": tool_call.id,
    #                     "role": "tool",
    #                     "name": function_name,
    #                     "content": function_response,
    #                     "text": f"Calling {function_name}"
    #                 }                
    #             )
    #         start = datetime.now()
    #         response = await LLM.client.chat.completions.create(
    #             model="gpt-4-1106-preview",
    #             messages=self.prompt_messages,
    #         )  # get a new response from the model where it can see the function response
    #         print('second call response:', response)
    #         log.info('Time for llm tool processing: %d', (datetime.now()-start).total_seconds())
    #         response_message = response.choices[0].message
    #     await self.appendMessage({
    #         "role": "assistant",
    #         "content": [
    #             {
    #                 "type": "text",
    #                 "text": response_message.content
    #             },
    #         ],
    #     })