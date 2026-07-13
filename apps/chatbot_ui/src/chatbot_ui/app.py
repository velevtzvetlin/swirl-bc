import streamlit as st
import requests
from chatbot_ui.core.config import config  # this is importing from the file we created src/core/config

# on every launch the layout wide
# initial sidebars state is expanded
st.set_page_config(
    page_title="Ecommerce Assitant",
    layout="wide",
    initial_sidebar_state="expanded",
)

def api_call(method, url, **kwargs):
    def _show_error_popup(message):
        """Show error message as a popup in the top-right corner."""
        st.session_state["error_popup"] = {
            "visible": True,
            "message": message
        }

    try:
        response = getattr(requests, method)(url, **kwargs)

        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            response_data = {"message": "Invalid response format from server"}

        if response.ok:
            return True, response_data

        return False, response_data

    except requests.exceptions.ConnectionError:
        _show_error_popup("Connection error. Please check your network connection.")
        return False, {"message": "Connection error"}
    except requests.exceptions.Timeout:
        _show_error_popup("The request timed out. Please try again later.")
        return False, {"message": "Request timeout"}
    except Exception as e:
        _show_error_popup(f"An unexpected error occurred: {str(e)}")
        return False, {"message": str(e)}

# with st.sidebar:
#     st.title("Settings")
    
#     provider = st.selectbox(
#         "Provider",
#         ["OpenAI", "Groq", "Google"]
#     )
    
#     if provider == "OpenAI":
#         model_name = st.selectbox(
#             "Model",
#             ["gpt-5-nano", "gpt-5-mini"]
#         )
#     elif provider == "Groq":
#         model_name = st.selectbox(
#             "Model",
#             ["llama-3.3-70b-versatile"]
#         )
#     else:
#         model_name = st.selectbox(
#             "Model",
#             ["gemini-2.5-flash"]
#         )
#     # store all kinds of varsa in the session_state object
#     st.session_state.provider = provider
#     st.session_state.model_name = model_name
    
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! How can I assist you today?"
    }]
if "used_context" not in st.session_state:
    # set as empty list every time we 
    st.session_state.used_context = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar:
    suggestions_tab, = st.tabs(["Suggestions"])
    with suggestions_tab:
        if st.session_state.used_context:
            for idx, item in enumerate(st.session_state.used_context):
                st.caption(item.get('description', 'No description'))
                if 'image_url' in item:
                    st.image(item['image_url'])
                st.caption(f"Price: {item.get('price')} USD")
                st.divider()
        else:
            st.info("No suggestions available.")


if prompt := st.chat_input("Hello! How can I assist you today?"):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        state,output = api_call(
            method="post",
            url=f"{config.API_URL}/agent",
            json={
                "query": prompt
            }
        )
        answer = output.get("answer", "")
        used_context = output.get("used_context", [])
        
        
        st.session_state.used_context = used_context
        
        st.write(answer)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        }) 
    st.rerun()
