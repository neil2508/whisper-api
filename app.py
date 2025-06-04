from flask import Flask, request, jsonify
from openai import OpenAI
import os
import tempfile

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    audio_file = request.files['file']

    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            audio_file.save(temp_file)
            temp_file_path = temp_file.name

        # Transcribe using OpenAI Whisper
        with open(temp_file_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )

        return jsonify({'transcription': transcript.text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.route('/', methods=['GET'])
def home():
    return 'Whisper transcription (OpenAI v1+) is running.'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
