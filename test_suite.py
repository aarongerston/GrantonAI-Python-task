
import os
import pytest
from main import app
from unittest.mock import patch, MagicMock
from ai import Categorizer, MissingEnvironmentVariable, VALID_MODELS


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


### API Tests ###

def test_categorize_text_success(client):
    """Test /api/categorize endpoint with valid text input."""
    with patch("ai.Categorizer.categorize_text", return_value="Technology"):
        response = client.post("/api/categorize", json={"text": "AI is advancing rapidly."})
    assert response.status_code == 200
    assert response.get_json() == "Technology"


def test_categorize_text_missing_text(client):
    """Test /api/categorize with missing text field."""
    response = client.post("/api/categorize", json={})
    assert response.status_code == 400
    assert response.get_json() == {"error": "Input text not supplied."}


### Categorizer Tests ###

def test_categorizer_init_missing_env():
    """Test Categorizer initialization fails when API key is missing."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(MissingEnvironmentVariable):
            Categorizer()


def test_categorizer_init_valid_model():
    """Test Categorizer initializes with a valid model."""
    with patch.dict(os.environ, {VALID_MODELS["gpt-3.5-turbo"]: "dummy_key"}):
        with patch("openai.Model", return_value=MagicMock()):
            bot = Categorizer()
            assert bot.model is not None


def test_categorizer_categorize_text():
    """Test categorize_text function with a mocked OpenAI response."""
    mock_model = MagicMock()
    mock_model.classify.return_value = {"choices": [{"text": "Sports"}]}

    with patch.dict(os.environ, {VALID_MODELS["gpt-3.5-turbo"]: "dummy_key"}):
        with patch("openai.Model", return_value=mock_model):
            bot = Categorizer()
            category = bot.categorize_text("Football is a popular sport.")
            assert category == "Sports"


def test_categorizer_categorize_text_invalid_response():
    """Test categorize_text function when the response is not in valid categories."""
    mock_model = MagicMock()
    mock_model.classify.return_value = {"choices": [{"text": "UnknownCategory"}]}

    with patch.dict(os.environ, {VALID_MODELS["gpt-3.5-turbo"]: "dummy_key"}):
        with patch("openai.Model", return_value=mock_model):
            bot = Categorizer()
            category = bot.categorize_text("Some random text.")
            assert category == "Failed to classify."
