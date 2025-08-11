import streamlit as st

def chatbot_response(user_input):
    user_input = user_input.lower()
    if 'hello' in user_input or 'hi' in user_input:
        return "Hello! How can I help you?"
    elif 'how are you' in user_input:
        return "I'm good, thanks for asking!"
    elif 'your name' in user_input:
        return "I'm a simple Python chatbot."
    else:
        return "Sorry, I don't understand that yet."

st.title("Simple AI Chatbot")

user_input = st.text_input("You:")

if user_input:
    response = chatbot_response(user_input)
    st.write("Chatbot:", response)
