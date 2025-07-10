import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")  # Recommended

class GPTFunctions:
    
    @staticmethod
    def get_knot_placement_advice():
        prompt = """
        I want to fit a cubic spline with 4 knots to the bootstrapped forward curve. 
        Where should I fit the 4 knots and why? Assume I have a plot showing a forward rate curve and market yields, with rates increasing steeply in early terms, peaking around 12 years, and flattening beyond 25 years.
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in financial data modeling."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )

        return response['choices'][0]['message']['content']