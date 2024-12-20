# from stt_whisper import startWhisper, stopWhisper
from pydub import AudioSegment
import numpy as np
from samplerate import resample
import datetime
import os
# w = startWhisper()


# Load MP3 file
# audio = AudioSegment.from_mp3('recordings/20241209-200703/fullcapture.mp3')
audio = AudioSegment.from_mp3('fullcapture.mp3')
print('audio:', audio)

# Get sample rate
sample_rate = audio.frame_rate

# Get the number of channels
num_channels = audio.channels

bytes_per_sample = audio.sample_width

# Print the sample rate and number of channels
print(f"Sample Rate: {sample_rate}")
print(f"Channels: {num_channels}")
print(f"Bytes per sample: {bytes_per_sample}")

frames_per_second = 50


#48000
frame_samples = sample_rate // frames_per_second
frame_size = frame_samples*num_channels
frame_bytes = frame_size * bytes_per_sample # 2 bytes per sample
print('frame_samples:', frame_samples)
print('frame_size:', frame_size, 'frame_bytes:', frame_bytes)


print('8 frames :', frame_size*8, 'frame bytes*8:', frame_bytes*8)

# print('audio:', audio._data)

# Extract chunks of data from a byte array n bytes at a time
n = frame_bytes
audio_data = audio.raw_data

print('total audio data:', len(audio_data))
print('total frames:', len(audio_data) // frame_bytes)
print('total seconds:', len(audio_data) // frame_bytes / frames_per_second) 

# audio_data = np.frombuffer(audio_data, dtype=np.int16)

frames = [np.frombuffer(audio_data[i:i + n], dtype=np.int16) for i in range(0, len(audio_data), n)]
# 20 ms frame
# 20 * 8 = 160 ms
# doing 8 at a time so that mono samples are divisible by 512 for silero
# silero needs 512 samples
# 160/5 = 32 ms silero resolution
frames8 = [frames[i:i+8] for i in range(0, len(frames), 8)]

# import os
# if not os.path.exists('processed'):
#     os.makedirs('processed',exist_ok=True)

# # Print the first few chunks to verify
# for i, chunk in enumerate(chunks8):
#     # print(f"Chunk {i}: {len(chunk)}")
#     w.send('prompt', [chunk])
#     w.inQ.put(('process',(chunk,'foo','processed')))




# stopWhisper()
import torch
from transformers import pipeline

print('Loading Silero VAD')
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                            model='silero_vad',
                            force_reload=True)
print('Loaded Silero VAD')

print('Loading Whisper')
asr = pipeline("automatic-speech-recognition",model="openai/whisper-base.en",device='cpu')
# asr = pipeline("automatic-speech-recognition",model="openai/whisper-small",device='cpu')
print('Starting Whisper',asr)

ratio = 16000 / 48000

t = 0 # ms
s = False
run = []
sCount = 0

#TODO should pad to 8 frames
# prev = None

def stt(buffer):
    buffer = np.concatenate((np.zeros(1024, dtype=np.int16), buffer))
    now = datetime.datetime.now()
    # filename = 
    if not os.path.exists('test'):
        os.makedirs('test')

    captureDir = 'test'
    filename = f'{captureDir}/{now.strftime("%Y%m%d-%H%M%S")}.mp3'
    audio_segment = AudioSegment(
        buffer.tobytes(), 
        frame_rate=16000,
        sample_width=2,  # 2 bytes for 16-bit audio
        channels=1
    )
    audio_segment.export(filename, format="mp3")

    float_buffer = buffer.astype(np.float32) / np.iinfo(np.int16).max
    return asr(float_buffer)

for f8 in frames8:
    buffer = buffer = np.concatenate(f8, axis=0)
    buffer = buffer[::2]
    buffer = resample(buffer, ratio, 'sinc_best')
    buffer = buffer.astype(np.int16)

    float_buffer = buffer.astype(np.float32) / np.iinfo(np.int16).max
    # stack up 512 samples
    # if len(float_buffer) % 512 != 0:
    #     print('bailing out not a multiple of 512')
    #     continue
    if len(float_buffer) % 512 != 0:
        pad_length = 512 - (len(float_buffer) % 512)
        float_buffer = np.pad(float_buffer, (0, pad_length), 'constant')
    float_buffer2 = float_buffer.reshape(-1,512)
    # print('float_buffer2:', float_buffer2.shape)

    p = model(torch.from_numpy(float_buffer2),16000)
    # print(p)
    p = torch.all(p<0.25)
    # p = torch.all(p<0.1)
    # p = False # no silence

    voice_detected = not p

    # print(p)
    if voice_detected:
        if not s:
            print('voice start: ', t)
            s = True
            sCount = 0
            run = []
            # if prev is not None:
            #     print('adding prev')
            #     run.append(prev)
        run.append(buffer)
    elif not voice_detected and s:
        print('voice end: ', t)
        s = False

    if not s:
        sCount = sCount + 1
        if sCount == 2 and len(run) > 0:
            print('silence: ', t)
            # TODO minimum run length
            print('process run:', len(run))
            buffer3 = np.concatenate(run, axis=0)
            print('buffer3:', buffer3.shape)
            # text = asr(buffer3)
            text = stt(buffer3)
            print(text)
            run = []

    # prev = float_buffer
    t = t + 160

if len(run) > 0:
    print('process run remnant:', len(run))
    buffer3 = np.concatenate(run, axis=0)
    text = stt(buffer3)
    print(text)
    run = []

# asr = pipeline("automatic-speech-recognition",model="openai/whisper-large-v3",device=0)
# asr = pipeline("automatic-speech-recognition",model="openai/whisper-tiny.en",device=0)
# print('Loading Whisper')
# asr = pipeline("automatic-speech-recognition",model="openai/whisper-base.en",device='cpu')
# print('Starting Whisper',asr)

ratio = 16000 / 48000

#TODO frames? or chunks?
buffer = buffer = np.concatenate(frames, axis=0)
buffer = buffer[::2]
buffer = resample(buffer, ratio, 'sinc_best')
buffer = buffer.astype(np.int16)

float_buffer = buffer.astype(np.float32) / np.iinfo(np.int16).max

t = asr(float_buffer)
print(t)

t = asr(float_buffer)
print(t)

t = asr(float_buffer)
print(t)