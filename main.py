
from flask import Flask, request, jsonify

from ai import Categorizer

app = Flask(__name__)


@app.route("/api/categorize", methods=["POST"])
def categorize_text():

    # Get input text from API request
    data = request.json
    text = data.get("text")
    if not text:
        return jsonify({"error": "Input text not supplied."}), 400

    # Initialize text classifier bot
    bot = Categorizer()

    # Use classifier bot to classify the given text
    category = bot.categorize_text(text)

    return jsonify(category), 200


if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
    )