import numpy as np
import pyaudio
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import deepspeech

class VoiceToText:
    def __init__(self, model_file_path, scorer_file_path):
        self.model = deepspeech.Model(model_file_path)
        self.model.enableExternalScorer(scorer_file_path)

    def record_and_transcribe(self, seconds=5):
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
            frames.append(np.frombuffer(data, dtype=np.int16))

        print("Finished recording")
        stream.stop_stream()
        stream.close()
        audio.terminate()

        buffer = np.concatenate(frames, axis=0)
        return self.model.stt(buffer)

class GPT2Response:
    def __init__(self, model_name="gpt2-medium"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate_response(self, input_text):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        response_ids = self.model.generate(input_ids)
        response = self.tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response

if __name__ == "__main__":
    # Voice to Text
    model_file_path = "path_to_your_model.pbmm"
    scorer_file_path = "path_to_your_scorer.scorer"
    voice_to_text = VoiceToText(model_file_path, scorer_file_path)
    transcribed_text = voice_to_text.record_and_transcribe()
    print("Transcribed Text:", transcribed_text)

    # GPT-2 response
    gpt = GPT2Response()
    response = gpt.generate_response(transcribed_text)
    print("GPT-2 Response:", response)
