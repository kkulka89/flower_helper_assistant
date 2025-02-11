import streamlit as st

from FlowerAssistant import ChatBot

bot = ChatBot()

st.set_page_config(page_title="Flowerbot2000")
with st.sidebar:
    st.title('Flowerbot2000')


# Function for generating LLM response
def generate_response(input):
    st.session_state
    result = bot.invoke(input)

    return result


# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm Flowerbot2000, your friendly flower assistant! "}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        print(input)
        st.write(input)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Getting your answer from mystery stuff.."):
            response = generate_response(input)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)