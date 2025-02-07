
import os
import string
from openai import OpenAI, OpenAIError, RateLimitError, APIConnectionError

CATEGORIES = (
    "Politics",
    "Sports",
    "Technology",
    "Other"
)

# valid models and the necessary API key env variable to use them
VALID_MODELS = {
    "gpt-3.5-turbo": "OpenAI",  # Not the best, but free
    # ...
}

ENV_VARS = {
    "OpenAI": "OPENAI_API_KEY",
    # ...
}


class MissingEnvironmentVariable(Exception):
    pass


class Categorizer:
    """
    LLM-based text classifier
    """

    def __init__(
            self,
            model: str = list(VALID_MODELS.keys())[0],
    ):

        self.model = model
        self.model_src = None
        self.llm = None

        self._init_llm()

    def _init_llm(self):
        """
        Initializes the specified LLM.
        Stores initialized LLM in self.llm

        :return: None
        """

        # Verify inputs...
        if self.model in VALID_MODELS.keys():
            self.model_src = VALID_MODELS[self.model]
            env_var = ENV_VARS[self.model_src]
            api_key = os.getenv(env_var)
            if api_key is None:
                raise MissingEnvironmentVariable(f"Your selected model `{self.model}` requires environment variable `{env_var}` that was not found.")
        else:
            raise ValueError("Invalid model.")

        # Initialize an OpenAI model
        if self.model_src == "OpenAI":
            llm = OpenAI(api_key=api_key)

        # Initialize another model...
        else:
            llm = None  # Model-specific initialization steps

        self.llm = llm

    def categorize_text(self, text: str):
        """
        Uses the given model to categorize text into categories defined in CATEGORIES.

        :param text: input text to categorize
        :return: (str) category or "Failed to classify."
        """

        prompt = (f"Classify the following text as one of the following categories: {', '.join(CATEGORIES)}."
                  f"Your answer should contain only the category and nothing else, not even punctuation.")

        if self.model_src == "OpenAI":
            try:
                completion = self.llm.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": text}
                    ]
                )
                output = completion.choices[0].message.content
            except OpenAIError as e:
                raise RuntimeError(f"OpenAI initialization error {str(e)}")
            except RateLimitError:
                raise RuntimeError("Sorry, you're out of OpenAI tokens! Pay up, bud.")
            except APIConnectionError:
                raise RuntimeError("Network issues. Try again.")  # Realistically would implement internal retry...
            except Exception as e:
                raise RuntimeError(f"OpenAI initialization error {str(e)}")

        else:
            # Use another LLM's API to classify text...
            output = ""

        # Clean the AI's response (remove any punctuation that the model rebelliously included and any whitespace)
        category = output.translate(str.maketrans("", "", string.punctuation)).strip()

        # Obviously some more advanced error checking should be done here to make sure that the output is as expected...
        if category in CATEGORIES:
            return category
        else:
            return "Failed to classify."
