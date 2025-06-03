from flask import Flask, request, jsonify
import subprocess
import json
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def try_fix_json(json_str):
    # Remove markdown-style code blocks
    json_str = json_str.strip()
    json_str = re.sub(r"```json|```", "", json_str)

    #replace curly quotes with standard ones
    json_str = json_str.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    #Counts opening and closing braces
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    if close_braces < open_braces:
        json_str += '}' * (open_braces - close_braces)

    #if the JSON doesn't end with a closing brace, add one
    if not json_str.strip().endswith('}'):
        json_str += '}'

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return None

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        prompt_input = data.get("prompt", "").strip()

        if not prompt_input:
            return jsonify({"error": "Empty prompt"}), 400

        # below lines are the strict prompt to LLaMA
        prompt = f"""
You are a strict JSON-only music recommendation engine.

Based on the following input: "{prompt_input}", return only a valid JSON object with exactly 5 song recommendations in this format:

{{
  "recommendations": [
    {{"title": "string", "artist": "string"}},
    {{"title": "string", "artist": "string"}},
    {{"title": "string", "artist": "string"}},
    {{"title": "string", "artist": "string"}},
    {{"title": "string", "artist": "string"}}
  ]
}}

Do not include explanations, markdown formatting, or any extra text. Only return valid JSON.
"""

        result = subprocess.run(#subprocess is used to call llama model via ollama
            ["ollama", "run", "llama3", prompt],
            capture_output=True,
            text=True,
            timeout=180,
            encoding='utf-8'
        )

        raw_output = result.stdout.strip()

        match = re.search(r'\{[\s\S]*\}', raw_output)#extracting json content from llama output
        if match:
            json_str = match.group(0)
            parsed = try_fix_json(json_str)
            if parsed:
                return jsonify(parsed)
            else:
                return jsonify({"error": "Invalid JSON even after fixes", "raw": raw_output}), 500
        else:
            return jsonify({"error": "No JSON found in model output", "raw": raw_output}), 500

    except subprocess.TimeoutExpired:
        return jsonify({"error": "LLaMA 3 timed out"}), 504
    except FileNotFoundError:
        return jsonify({"error": "Ollama not installed or not in PATH"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
