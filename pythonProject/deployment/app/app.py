import streamlit as st
import requests

FASTAPI_URL = "http://127.0.0.1:8000/predict/"

def main():
    st.title("Software Requirements Extractor")
    st.write("Use this app to extract technical requirements from job descriptions or related texts.")

    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "prediction" not in st.session_state:
        st.session_state.prediction = ""

    st.session_state.user_input = st.text_area(
        "Enter the text to analyze:",
        height=200,
        value=st.session_state.user_input,
        placeholder="Type your text here..."
    )

    if st.button("Extract Requirements"):
        if st.session_state.user_input.strip():
            st.session_state.prediction = send_request_and_display_result(st.session_state.user_input)
        else:
            st.warning("Please enter some text to analyze.")

    st.text_area("Extracted Requirements:", value=st.session_state.prediction, height=200, disabled=True)

def send_request_and_display_result(user_input):
    with st.spinner("Extracting requirements..."):
        try:
            response = requests.post(FASTAPI_URL, json={"text": user_input})

            if response.status_code == 200:
                return response.json().get("predicted_requirements", "No prediction returned.")
            else:
                st.error(f"Error {response.status_code}: {response.json().get('detail', 'Unknown error')}.")
                return ""

        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")
            return ""

if __name__ == "__main__":
    main()
