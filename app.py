from flask import Flask, request, jsonify
import whisper
import os

app = Flask(__name__)
model = whisper.load_model("base")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    audio_file = request.files['file']
    filepath = f"temp_audio.{audio_file.filename.split('.')[-1]}"
    audio_file.save(filepath)

    try:
        result = model.transcribe(filepath)
        return jsonify({'transcription': result["text"]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(filepath)

@app.route('/', methods=['GET'])
def home():
    return 'Whisper transcription server is running.'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
