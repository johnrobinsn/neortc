import threading
import asyncio

from tts_kokoro_sync import TTS_Kokoro
from threadworker import ThreadWorker

# Async friendly wrapper for TTS_Kokoro
# non-blocking API for doing text-to-speech using the kokoro model.

class Async_TTS_Kokoro(TTS_Kokoro):
    def __init__(self, opus_frame_handler=None):

        async def _invoke_frame_handler_on_main_thread(opus_frame):
            if asyncio.iscoroutinefunction(self.opus_frame_handler):
                await self.opus_frame_handler(opus_frame)
            else:
                self.opus_frame_handler(opus_frame)

        def _threaded_opus_frame_handler(opus_frame):
            # if self.opus_frame_handler:
            #     asyncio.run_coroutine_threadsafe(_invoke_frame_handler_on_main_thread(opus_frame), self.loop)  
            # if self.opus_frame_handler:
            #     self.loop.call_soon_threadsafe(self.opus_frame_handler, opus_frame)
            if self.opus_frame_handler:
                self.opus_frame_handler(opus_frame)

        super().__init__(opus_frame_handler=_threaded_opus_frame_handler)
        self.w = ThreadWorker(supportOutQ=False)
        self.shared_lock = threading.Lock()
        self.loop = asyncio.get_event_loop()
        self.opus_frame_handler = opus_frame_handler

    def super_say(self, t):
        with self.shared_lock:
            return super().say(t)
        
    async def say(self,t):
        self.w.add_task(self.super_say, t)

## ----------------------------

def main():
    import aiofiles
    import sys
    from termcolor import colored

    # load_dotenv(".env.local")

    from tts_utils import OpusStreamPlayer

    opus_player = OpusStreamPlayer()

    def opus_frame_handler(opus_frame):
        opus_player.write(opus_frame)      

    print(colored(f'Async_TTS_Kokoro', 'cyan'))

    tts = Async_TTS_Kokoro(opus_frame_handler=opus_frame_handler)

    async def read_lines():

        await tts.say('The sky above the port was the color of television, tuned to a dead channel.')
        await tts.say('hello world hello world hello world')    

        async with aiofiles.open('/dev/stdin', mode='r') as f:
            print(colored("> ","yellow"), end="")
            sys.stdout.flush()
            async for line in f:    
                # print(colored(await askit.prompt1(line.strip(), moreTools=[get_current_location]),"green"))
                await tts.say(line.strip())
                print(colored("> ","yellow"), end="")
                sys.stdout.flush()

    asyncio.run(read_lines())

if __name__ == '__main__':
    main()