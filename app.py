from flask import Flask, request, jsonify
from openai import OpenAI
import os

app = Flask(__name__)

# Initialize the OpenAI client using your API key from environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    audio_file = request.files['file']

    try:
        # Pass the uploaded file directly, not .stream
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return jsonify({'transcription': transcript.text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return 'Whisper transcription (OpenAI v1+) is running.'

if __name__ == '__main__':
    # Start the Flask app on port 5000, accessible from all interfaces
    app.run(host='0.0.0.0', port=5000)

