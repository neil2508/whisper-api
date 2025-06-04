from flask import Flask, request, jsonify
import openai
import os

app = Flask(__name__)

# Set your API key securely (see below)
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    audio_file = request.files['file']
    try:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        return jsonify({'transcription': transcript['text']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return 'Whisper API (OpenAI) is running.'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
