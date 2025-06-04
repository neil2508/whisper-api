from flask import Flask, request, jsonify
from openai import OpenAI
import os

app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    audio_file = request.files['file']

    try:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file.stream
        )
        return jsonify({'transcription': transcript.text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return 'Whisper transcription (OpenAI v1+) is running.'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

