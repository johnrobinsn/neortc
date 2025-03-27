import numpy as np

from kokoro import KPipeline
from samplerate import resample
from pyogg import OpusDecoder,OpusEncoder

# import soundfile as sf

class TTS_Kokoro:
    def __init__(self, opus_frame_handler=None):
        self.kokoro_pipeline = KPipeline(lang_code='a') # <= make sure lang_code matches voice
        self.kokoro_sample_rate = 24000
        self.kokoro_channels = 1

        self.opus_encoder = OpusEncoder()
        self.opus_encoder.set_sampling_frequency(48000)
        self.opus_encoder.set_channels(1)
        self.opus_encoder.set_application('audio')

        self.opus_frame_handler = opus_frame_handler  

    def say(self,t):
        generator = self.kokoro_pipeline(
            [t], voice='af_heart', # <= change voice here
            speed=1, split_pattern=r'\n+'
        )

        for i, (_gs, _ps, audio) in enumerate(generator):
            break

        # sf.write(f'{i}.wav', audio, 24000) # save each audio file

        samples_per_second = self.kokoro_sample_rate # samplerate from model

        # resample the audio to 48kHz
        target_sample_rate = 48000 # pyogg friendly sample rate
        ratio = target_sample_rate / samples_per_second
        audio = resample(audio, ratio, 'sinc_best')
        samples_per_second = target_sample_rate

        audio16 = (audio*32767).astype(np.int16)  # scale and convert to int16

        frame_size = int(samples_per_second * self.kokoro_channels * 20/1000) # 20ms at 48kHz
        for i in range(0, len(audio16), frame_size):
            pcm_frame = audio16[i:i+frame_size].tobytes()
            # pad with zeros if needed
            if len(pcm_frame) < frame_size * 2:  # 2 bytes per sample for int16
                pcm_frame = pcm_frame.ljust(frame_size * 2, b'\x00')
            opus_frame = bytearray(self.opus_encoder.encode(audio16[i:i+frame_size].tobytes()))  # Get raw Opus frame
            if self.opus_frame_handler:
                self.opus_frame_handler(opus_frame)


if __name__ == "__main__":
    from tts_utils import OpusStreamPlayer

    opus_player = OpusStreamPlayer()

    def opus_frame_handler(opus_frame):
        opus_player.write(opus_frame)      

    tts = TTS_Kokoro(opus_frame_handler=opus_frame_handler)
    tts.say('The sky above the port was the color of television, tuned to a dead channel.')
    tts.say('hello world hello world hello world')

    while True:
        user_input = input(">")
        if user_input.lower() == 'exit':
            break
        tts.say(user_input)
