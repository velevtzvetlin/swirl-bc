import uuid

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

def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4()) # generate a unique session id for each user
    return st.session_state.session_id

thread_id = get_session_id() # in the future we can generate unique thread ids for each conversation and pass that to the backend to keep track of conversation history if we want to implement multi-turn conversations

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

def submit_feedback(feedback_type=None, feedback_text=""):
    """Submit feedback to the API endpoint"""

    def _feedback_score(feedback_type):
        if feedback_type == "positive":
            return 1
        elif feedback_type == "negative":
            return 0
        else:
            return None

    feedback_data = {
        "feedback_score": _feedback_score(feedback_type),
        "feedback_text": feedback_text,
        "trace_id": st.session_state.trace_id,
        "feedback_source_type": "api"
    }

    status, response = api_call("post", f"{config.API_URL}/submit_feedback", json=feedback_data)
    return status, response
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
    
if "thread_id" not in st.session_state:
    st.session_state.thread_id = thread_id

if "trace_id" not in st.session_state:
    st.session_state.trace_id = ""
    
if "latest_feedback" not in st.session_state:
    st.session_state.latest_feedback = None  # can be "positive" or "negative"
    
if "show_feedback_box" not in st.session_state:
    st.session_state.show_feedback_box = False  # controls visibility of the feedback text box
    
if "feedback_submission_status" not in st.session_state:
    st.session_state.feedback_submission_status = None  # can be "success" or "error"

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

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
            
for idx, message in enumerate[dict[str, str]](st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

    # Add feedback buttons only for the latest assistant message (excluding the initial greeting
    is_latest_assistant = (
        message["role"] == "assistant" and
        idx == len(st.session_state.messages) - 1 and
        idx > 0
    )

    if is_latest_assistant:
        # Use Streamlit's built-in feedback component
        feedback_key = f"feedback_{len(st.session_state.messages)}"
        feedback_result = st.feedback("thumbs", key=feedback_key)

        # Handle feedback selection
        if feedback_result is not None:
            feedback_type = "positive" if feedback_result == 1 else "negative"

            # Only submit if this is a new/different feedback
            if st.session_state.latest_feedback != feedback_type:
                with st.spinner("Submitting feedback..."):
                    status, response = submit_feedback(feedback_type=feedback_type)
                if status:
                    st.session_state.latest_feedback = feedback_type
                    st.session_state.feedback_submission_status = "success"
                    st.session_state.show_feedback_box = (feedback_type == "negative")
                else:
                   st.session_state.feedback_submission_status = "error"
                   st.error("Failed to submit feedback. Please try again.")
                   st.rerun()

        # Show feedback status message
        if st.session_state.latest_feedback and st.session_state.feedback_submission_status == "success":
            if st.session_state.latest_feedback == "positive":
                st.success("✅ Thank you for your positive feedback!")
            elif st.session_state.latest_feedback == "negative" and not st.session_state.show_feedback_box:
                st.success("✅ Thank you for your feedback!")
        elif st.session_state.feedback_submission_status == "error":
            st.error("❌ Failed to submit feedback. Please try again.")

# Show feedback text box if thumbs down was pressed
        if st.session_state.show_feedback_box:
                    st.markdown("**Want to tell us more? (Optional)**")
                    st.caption("Your negative feedback has already been recorded. You can optionally provide additional")
        
        feedback_text = st.text_area(
            "Additional feedback (optional)",
            key=f"feedback_text_{len(st.session_state.messages)}",
            placeholder="Please describe what was wrong with this response...",
            height=100
        )
        
        
        col_send, col_spacer, col_close = st.columns([3, 5, 2])
        with col_send:
            if st.button("Send Additional Details", key=f"send_additional_{len(st.session_state.messages)}"):
                if feedback_text.strip(): # Only send if there's actual text
                    with st.spinner("Submitting additional feedback..."):
                        status, response = submit_feedback(feedback_text=feedback_text)
                    if status:
                        st.success("✅ Thank you! Your additional feedback has been recorded.")
                        st.session_state.show_feedback_box = False
                    else:
                        st.error("❌ Failed to submit additional feedback. Please try again.")
                else:
                    st.warning("Please enter some feedback text before submitting.")
                st.rerun()
        with col_close:
            if st.button("Close", key=f"close_feedback_{len(st.session_state.messages)}"):
                st.session_state.show_feedback_box = False
                st.rerun()

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
                "query": prompt,
                "thread_id": st.session_state.thread_id
            }
        )
        answer = output.get("answer", "")
        used_context = output.get("used_context", [])
        trace_id = output.get("trace_id", "")
        st.session_state.trace_id = trace_id
        st.session_state.used_context = used_context
        
        st.write(answer)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        }) 
    st.rerun()
