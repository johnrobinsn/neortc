import stt_whisper
from pydub import AudioSegment

# startWhisper()


# Load MP3 file
audio = AudioSegment.from_mp3('recordings/20241209-200703/fullcapture.mp3')
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

frames_per_second = sample_rate/25


#48000
frame_size = frames_per_second*num_channels*bytes_per_sample # 2 channels, 2 bytes per sample
print('frame_size:', frame_size)

# print('audio:', audio._data)


# stopWhisper()