from transformers import pipeline

class LLMModel:
    def __init__(self):
        # Use a generative model instead of extractive QA
        self.generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base"
        )

    def generate_response(self, question, context="", system_message="You are a helpful assistant."):
        try:
            prompt = f"{system_message}\nContext: {context}\nQuestion: {question}\nAnswer:"
            result = self.generator(prompt, max_new_tokens=100)
            return result[0]["generated_text"]
        except Exception as e:
            return f"Error: {str(e)}"