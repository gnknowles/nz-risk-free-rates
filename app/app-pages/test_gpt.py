import streamlit as st
import os
from openai import OpenAI

st.title("🔌 ChatGPT API Test with New SDK")

# Use environment variable

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY is not set. Please set the environment variable.")
else:
    client = OpenAI(api_key=api_key)

    prompt = st.text_input("Ask ChatGPT something:", value="Say hello!")

    if st.button("Send to ChatGPT"):
        try:
            with st.spinner("Waiting for response..."):
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                st.success("Response received!")
                st.markdown("**ChatGPT says:**")
                st.write(response.choices[0].message.content)

        except Exception as e:
            st.error(f"Error: {e}")
