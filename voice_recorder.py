import pyaudio
import numpy as np
import wave

def record_audio(seconds=5):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK_SIZE = 1024

    audio = pyaudio.PyAudio()
    print("Recording...")
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)
    frames = []

    for _ in range(0, int(RATE / CHUNK_SIZE * seconds)):
        data = stream.read(CHUNK_SIZE)
        frames.append(data)

    print("Finished recording")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open('recorded_audio.wav', 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

if __name__ == "__main__":
    record_audio()
