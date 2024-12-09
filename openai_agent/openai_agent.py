from openaiasync import prompt,promptStreaming

# startWorker()

from agent import startAgent

startAgent(prompt,'openai',promptStreaming)