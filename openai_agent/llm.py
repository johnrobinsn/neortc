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

from aconfig import config
import uuid

from abc import ABC, abstractmethod

import lancedb
from lancedb.rerankers import ColbertReranker


from pydantic import create_model
import inspect, json
from inspect import Parameter

def sums(a:int, b:int=1):
    "Adds a + b"
    return a + b

def schema(f):
    kw = {n:(o.annotation, ... if o.default==Parameter.empty else o.default)
          for n,o in inspect.signature(f).parameters.items()}
    s = create_model(f'Input for `{f.__name__}`', **kw).schema()
    return dict(name=f.__name__, description=f.__doc__, parameters=s)


print('schema:',schema(sums))

openai_api_key = config.get('openai_api_key')

db_path = os.path.expanduser('~/.neodocs.db')
db = lancedb.connect(db_path)
tbl = db.open_table("my_table")
reranker = ColbertReranker()

# results = (tbl.search(query, query_type="hybrid") # Hybrid means text + vector
#     # .where("category = 'film'", prefilter=True) # Restrict to only docs in the 'film' category
#     .limit(limit) # Get 10 results from first-pass retrieval
#     .rerank(reranker=reranker) # For the reranker to compute the final ranking
# )


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
    def __init__(self,promptFunc,agentName,promptStreamingFunc=None,initialPrompt=None):
        self.agentName = agentName
        self.promptFunc = promptFunc
        self.promptStreamingFunc = promptStreamingFunc
        self.created = datetime.utcnow().isoformat() + "Z"
        self.modified = datetime.utcnow().isoformat() + "Z"
        self.id = str(uuid.uuid4())
        self.name = ''
        self.summary = ''
        self.persisted = False
        self.dead = False
        LLM.nameIndex += 1
        self.currentEntry = None

        self.local_dt = datetime.utcnow() #now(datetime.timezone.utc)
        self.date_time_string = self.local_dt.strftime("%A, %B %d, %Y %H:%M:%S %Z UTC")
        if initialPrompt:
            self.prompt_messages = initialPrompt
        else:
            self.prompt_messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": f"You are a helpful AI assistant.  The current date and time is {self.date_time_string}.  When reporting the time or date, speak succinctly.  When telling a joke, put the whole joke on the first line. If I ask you to stop simply respond with the word \"OK\"."
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
            'modified':self.modified,
            'log':self.prompt_messages,
        }
        dirName = f'{self.agentName}_chats'
        os.makedirs(dirName, exist_ok=True)
        with open(f'{dirName}/{self.id}.json', 'w') as file:
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
        return {'id':self.id,'display':display,'summary':self.summary,'name':self.name,'created':self.created,'modified':self.modified}
    
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

    async def bargeIn(self):
        for l in self.listeners:
            await l(None)    

    def addMetaDatalistener(self,l):
        if l:
            self.metaDataListeners.append(l)
            # self.notifyMetaDataChanged()

    def delMetaDataListener(self,l):
        if l in self.metaDataListeners:
            self.metaDataListeners.remove(l)

    # user,assistant,system
    async def appendMessage(self,m):
        self.modified = datetime.utcnow().isoformat() + "Z"
        self.prompt_messages.append(m)
        # for l in self.listeners:
        #     await l(m if isinstance(m, dict) else m.__dict__)
        
        # ask model to summarize conversation
        summary = await self.summarize()
        print("Summary:", summary)
        # if self.summary != summary:
        self.summary = summary
        # always notify because the log has been modified (sort order etc
        await self.notifyMetaDataChanged()
        self.save()

    async def openEntry(self, role):
        self.currentEntry = {'role':role,'content':''}
        for l in self.listeners:
            await l({'t':'openEntry','role':role})
    
    async def writeEntry(self,data):
        self.currentEntry['content'] += data
        for l in self.listeners:
            await l({'t':'writeEntry','data':data,'role':self.currentEntry['role']})

    # content key vs data key... 
    async def closeEntry(self):
        for l in self.listeners:
            await l({'t':'closeEntry','role':self.currentEntry['role'],'data':self.currentEntry['content']})        
        await self.appendMessage({
                "role": self.currentEntry['role'],
                "content": [
                    {
                        "type": "text",
                        "text": self.currentEntry['content']
                    },
                ],
            })
        self.currentEntry = None

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
        return await self.promptFunc(log)             

    async def prompt(self,t):
        # await self.appendMessage({
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "text",
        #                 "text": t
        #             },
        #         ],
        #     })


        # enable rag 
        if False:
            results = (tbl.search(t, query_type="hybrid") # Hybrid means text + vector
                # .where("category = 'film'", prefilter=True) # Restrict to only docs in the 'film' category
                .limit(3) # Get 10 results from first-pass retrieval
                .rerank(reranker=reranker) # For the reranker to compute the final ranking
            )

            context = [e['text'] for e in results.to_list()]

            context = '\n\n'.join(context)

            t = f'''
Answer the following question using the provided context:

## Context
{context}

## Question
{t}
'''

        await self.openEntry('user')
        await self.writeEntry(t)
        await self.closeEntry()

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
        if self.promptStreamingFunc:
            response = self.promptStreamingFunc(self.prompt_messages)
            await self.openEntry('assistant')
            async for r in response:
                await self.writeEntry(r)
            await self.closeEntry()
        else:
            response = await self.promptFunc(self.prompt_messages)
            await self.openEntry('assistant')
            await self.writeEntry(response)
            await self.closeEntry()
        # response_message = response[0][0]['generated_text'][-1]['content']
        #openai
        # response = await LLM.client.chat.completions.create(
        #     model="gpt-4-1106-preview",
        #     messages=log,
        # )
        # response_message = response.choices[0].message.content

        # await self.appendMessage({
        #     "role": "assistant",
        #     "content": [
        #         {
        #             "type": "text",
        #             "text": response_message
        #         },
        #     ],
        # })        



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