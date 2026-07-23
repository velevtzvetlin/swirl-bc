from langsmith import Client

client = Client()

def submit_feedback(trace_id: str, feedback_score: int = None, feedback_text: str = "", feedback_source_type: str = "api"):
    """
    Submits feedback to the LangSmith API.

    Args:
        trace_id (str): The trace ID associated with the feedback.
        feedback_score (int): The feedback score (0 or 1).
        feedback_text (str): The feedback text.
        feedback_source_type (str): The source type of the feedback ('api' or 'user').
    """    
    if feedback_score is not None:
        client.create_feedback(
            run_id=trace_id,
            key="thumbs",
            score=feedback_score,
            feedback_source_type=feedback_source_type
        )
    
    if len(feedback_text) > 0:
        client.create_feedback(
            run_id=trace_id,
            key="comment",
            text=feedback_text,
            feedback_source_type=feedback_source_type
        )
    
    