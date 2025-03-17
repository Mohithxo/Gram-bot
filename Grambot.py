!pip install numpy
!pip install gtts
!pip install openai==0.28
!pip install kaleido
!pip install cohere

!pip install typing_extensions

!pip install -q openai-whisper
!pip install -q gradio
!apt-get install python3-pyaudio
!pip install SpeechRecognition
!pip install pyaudio
!pip install pydub
!pip install language-tool-python

import whisper
import gradio as gr
import openai
from gtts import gTTS
import speech_recognition as sr

model = whisper.load_model("base")
openai.api_key = 'xxxxxxxxxxxxxxxxx'

text_to_speak = "Hello, this is a sample text to convert to speech."

# Create gTTS object
tts = gTTS(text=text_to_speak, lang='en')

# Save the speech as a WAV file
tts.save("output1.wav")

from IPython.display import Audio, display

display(Audio('output1.wav', autoplay=True))

from transformers import pipeline

p = pipeline("automatic-speech-recognition", model="Santhosh-kumar/ASR")

import gradio as gr
from transformers import pipeline
import numpy as np

transcriber = pipeline("automatic-speech-recognition", model="Santhosh-kumar/ASR")

def transcribe(audio):
    if audio == None:
      return ["Try Again","Try Again","output.wav"]
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    input_text = transcriber({"sampling_rate": sr, "raw": y})["text"]

    messages = [
    {"role": "system", "content": "You are a helpful assistant."}]

    if input_text:
        messages.append(
            {"role": "user", "content": "correct the english in the following sentences."+input_text},
        )
        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
    reply = chat_completion.choices[0].message.content
    tts = gTTS(text=reply, lang='en')
    # Save the speech as a WAV file
    tts.save("output.wav")
    return(input_text,reply, 'output.wav')

from IPython.display import Javascript
from google.colab import output
from base64 import b64decode
from gtts import gTTS #Import Google Text to Speech
from IPython.display import Audio

RECORD = """
const sleep  = time => new Promise(resolve => setTimeout(resolve, time))
const b2text = blob => new Promise(resolve => {
  const reader = new FileReader()
  reader.onloadend = e => resolve(e.srcElement.result)
  reader.readAsDataURL(blob)
})
var record = time => new Promise(async resolve => {
  stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  recorder = new MediaRecorder(stream)
  chunks = []
  recorder.ondataavailable = e => chunks.push(e.data)
  recorder.start()
  await sleep(time)
  recorder.onstop = async ()=>{
    blob = new Blob(chunks)
    text = await b2text(blob)
    resolve(text)
  }
  recorder.stop()
})
"""

def record(sec=5):
  display(Javascript(RECORD))
  s = output.eval_js('record(%d)' % (sec*1000))
  b = b64decode(s.split(',')[1])
  with open('output.webm','wb') as f:
    f.write(b)
  return 'output.webm'

record()

display(Audio('output.webm', autoplay=True))

from pydub import AudioSegment
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the grammar correction model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")

# Function to correct grammar using the loaded model
def correct_grammar(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

# Convert WebM to WAV (if needed)
audio = AudioSegment.from_file('output.webm', format="webm")
audio.export('output.wav', format="wav")

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Load the WAV audio file and perform speech recognition
with sr.AudioFile('output.wav') as source:
    # Adjust for ambient noise and record the audio
    recognizer.adjust_for_ambient_noise(source)
    audio_data = recognizer.record(source)

    try:
        # Perform speech recognition using Google API
        text1 = recognizer.recognize_google(audio_data)
        print("Text from audio:", text1)

        # Correct grammar
        corrected_text = correct_grammar(text1)
        print("Corrected Text:", corrected_text)

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

text_to_speak = "Hello, this is a sample text to convert to speech."

# Create gTTS object
correct_text_speech = "the corrected text is."+corrected_text
tts = gTTS(correct_text_speech, lang='en')

# Save the speech as a WAV file
tts.save("output1.wav")

from IPython.display import Audio, display

display(Audio('output1.wav', autoplay=True))

import gradio as gr
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
from pydub import AudioSegment
import speech_recognition as sr

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Load the grammar correction model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")

# Function to transcribe and correct grammar
def transcribe_and_correct(audio):
    # Convert audio input to the correct format (WAV)
    audio_segment = AudioSegment.from_file(audio)  # Gradio passes audio as a temporary file
    audio_segment.export('output.wav', format="wav")

    # Load the WAV audio file and perform speech recognition
    with sr.AudioFile('output.wav') as source:
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.record(source)
        return "output.wav"

        try:
            # Perform speech recognition using Google API
            transcribed_text = recognizer.recognize_google(audio_data)

            # Correct grammar
            inputs = tokenizer(transcribed_text, return_tensors="pt", padding=True, truncation=True)
            outputs = model.generate(**inputs)
            corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Convert corrected text to speech using gTTS
            tts = gTTS(corrected_text, lang='en')
            tts.save("corrected_output.wav")  # Save corrected output as WAV

            # Return transcribed text, corrected text, and the corrected speech file
            return transcribed_text, corrected_text, "corrected_output.wav"

        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand the audio", "", ""
        except sr.RequestError as e:
            return f"Could not request results; {e}", "", ""
        except Exception as e:
            return f"An error occurred: {e}", "", ""

# Gradio interface
interface = gr.Interface(
    fn=transcribe_and_correct,
    inputs=gr.Audio(type="filepath"),  # Capture audio from the microphone or file input
    outputs=[
        gr.Textbox(label="Transcribed Text"),
        gr.Textbox(label="Corrected Text"),
        gr.Audio(label="Corrected Speech")  # Play the corrected speech audio
    ],
    title="Speech to Text & Grammar Correction",
    description="Record or upload audio, get transcribed text, corrected text, and the corrected sentence as audio."
)

# Launch the Gradio interface
interface.launch()
