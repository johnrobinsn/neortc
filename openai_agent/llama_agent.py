from llama3async import startWorker, prompt

startWorker()

from agent import startAgent

startAgent(prompt,'llama')