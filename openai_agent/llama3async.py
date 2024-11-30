
import logging
#logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

from multiprocessing import Process, Queue
import asyncio
# import transformers
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,pipeline

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

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

messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a pirate chatbot who always responds in pirate speak!"
                    },                    
                ],
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": "Who was George Washington?"
                    },   
                ]
            }
        ]



# # outputs = pipeline(
# #     messages,
# #     max_new_tokens=256,
# # )
# print(outputs[0]["generated_text"][-1])


class Worker:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.inQ = Queue()
        self.outQ = Queue()
        self.p = Process(
            target=self.loop,
            args=(self.inQ, self.outQ))
        self.p.start()

    @staticmethod
    def loop(inQ: Queue, outQ: Queue):
        try:
            # asr = pipeline("automatic-speech-recognition",model="openai/whisper-large-v3",device=0)
            # asr = pipeline("automatic-speech-recognition",model="openai/whisper-tiny.en",device=0)
            # asr = pipeline("automatic-speech-recognition",model="openai/whisper-tiny",device='cpu')
            

            # model_name = "meta-llama/Llama-3.1-8B"
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                load_in_8bit=True,
                device_map="auto"
            )

            tokenizer = AutoTokenizer.from_pretrained(model_id)            
            # pl = pipeline(
            #     "text-generation",
            #     model=model_id,
            #     #model_kwargs={"torch_dtype": torch.bfloat16},
            #     model_kwargs={"torch_dtype": torch.float16},
            #     device_map="auto",
            # )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Model loaded on device: {device}")

            pl = pipeline(
                task="text-generation",
                model=model,
                tokenizer=tokenizer
            )            
            
            print('Starting Worker',pipeline)

            while True:
                command, args = inQ.get()
                print('Command:',command)
                if command == 'stop':
                    break
                elif command == 'prompt':
                    #outQ.put((asr(args[0])))
                    # print('args:',args)
                    o = pl(args,max_new_tokens=256)
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

def startWorker():
    global w
    w = Worker()

async def prompt(messages):
    # print('Prompting')
    print('Prompting1:',messages)
    messages = [{"role":r['role'],"content":r['content'][0]['text']} for r in messages]
    print('Prompting2:',messages)
    r = (await w.send('prompt',(messages,)))[0][0]['generated_text'][-1]['content']
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