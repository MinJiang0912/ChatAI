from transformers import GPT2Tokenizer, GPT2LMHeadModel
import deepspeech
import numpy as np
import wave

class VoiceToText:
    def __init__(self, model_file_path, scorer_file_path):
        self.model = deepspeech.Model(model_file_path)
        self.model.enableExternalScorer(scorer_file_path)

    def transcribe(self, audio_buffer):
        return self.model.stt(audio_buffer)

class GPT2Response:
    def __init__(self, model_name="gpt2-medium"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate_response(self, input_text):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        response_ids = self.model.generate(input_ids)
        response = self.tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response

def load_audio():
    with wave.open('recorded_audio.wav', 'rb') as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

if __name__ == "__main__":
    # Load the recorded audio data from a file
    recorded_audio = load_audio()

    # Voice to Text
    model_file_path = "path_to_your_model.pbmm"
    scorer_file_path = "path_to_your_scorer.scorer"
    voice_to_text = VoiceToText(model_file_path, scorer_file_path)
    transcribed_text = voice_to_text.transcribe(recorded_audio)
    print("Transcribed Text:", transcribed_text)

    # GPT-2 response
    gpt = GPT2Response()
    response = gpt.generate_response(transcribed_text)
    print("GPT-2 Response:", response)
