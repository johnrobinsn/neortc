from pyogg import OpusBufferedEncoder,OggOpusWriter,OpusDecoder,OpusEncoder
import pyaudio

class OpusStreamPlayer:
    def __init__(self, channels=1):
        self.pa = pyaudio.PyAudio()
        self.pcm_stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=48000, output=True)
        self.opus_decoder = OpusDecoder()
        self.opus_decoder.set_channels(channels)
        self.opus_decoder.set_sampling_frequency(48000)        

    def __del__(self):
        # self.pcm_stream.stop_stream()
        # self.pcm_stream.close()
        # self.pa.terminate()
        pass

    def write(self, opus_frame):
        pcm_frame = self.opus_decoder.decode(opus_frame)
        self._writepcm(pcm_frame)

    def _writepcm(self, pcm):
        # chunk_size = 2048
        # for i in range(0, len(pcm), chunk_size):
        #     self.pcm_stream.write(pcm[i:i + chunk_size].tobytes())        
        self.pcm_stream.write(pcm.tobytes())
