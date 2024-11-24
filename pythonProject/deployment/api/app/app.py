import streamlit as st
import requests

# Title of the web app
st.title("Requirements Extractor")
# Subtitle
st.write("Insert here any unclear requirements that the customer sent you and we will try to decipher for you what he meant.")

requirements = st.text_input("Enter text to translate:", placeholder="Type your text here...")

if st.button("Translate"):
    if requirements.strip() == "":
        st.warning("Please enter some text to translate.")
    else:
        # Show a spinner while "processing"
        with st.spinner("Processing..."):
            # Simulating backend interaction (replace this with your backend logic)
            response = requests.post("http://fastapi-backend:8000/predict_answer/", requirements)

            if response.status_code == 200:
                result = response.json()
                translated_text = result.get("predicted_digit")
                st.success("Translation complete!")
                st.text_area("Translated Text", translated_text, height=100)
            else:
                st.error(f"Error: {response.text}")

