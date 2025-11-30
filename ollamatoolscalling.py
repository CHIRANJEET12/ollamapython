import ollama
import re
import streamlit as st



# --- Core Functions (from your original script) ---


def classify_issue(message: str) -> str:
    """
    Classify a support message into one of the categories based on keywords.
    """
    message = message.lower()
    if "refund" in message or "money" in message:
        return "Billing/Refund Issue"
    elif "login" in message or "password" in message:
        return "Account/Authentication Issue"
    elif "delivery" in message or "shipment" in message or "arrived" in message:
        return "Delivery/Logistics Issue"
    elif "error" in message or "not working" in message or "doesn't work" in message:
        return "Technical Issue"
    else:
        return "General Inquiry"

def analyze_sentiment(message: str) -> str:
    """
    Simple rule-based sentiment analysis to determine if a message is
    Positive, Negative, or Neutral.
    """
    negative_words = ["angry", "bad", "terrible", "not happy", "upset", "horrible", "frustrated"]
    positive_words = ["good", "great", "happy", "love", "excellent", "awesome"]
    text = message.lower()

    if any(w in text for w in negative_words):
        return "Negative"
    elif any(w in text for w in positive_words):
        return "Positive"
    else:
        return "Neutral"
    

# --- Tool Definitions for the Ollama Model ---


tools = [
    {
        "type": "function",
        "function": {
            "name": "classify_issue",
            "description": "Classify a customer support message into predefined issue categories like Billing, Account, Delivery, or Technical.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The customer's message text"},
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_sentiment",
            "description": "Analyze the sentiment of a customer message to determine if it is Positive, Neutral, or Negative.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The customer's message text"},
                },
                "required": ["message"],
            },
        },
    },
]


# --- Streamlit Application UI ---

st.set_page_config(layout="wide")
st.title("🤖 Customer Support Message Analyzer")
st.markdown(
    "This app uses a local LLM with function calling to classify a customer issue, analyze its sentiment, and generate a summary.")

customer_message = st.text_area(
    "Enter the customer's message below:",
    height=150
)

if st.button("Analyze Message", type="primary"):
    if not customer_message.strip():
        st.warning("Please enter a message to analyze.")
    else:
        try:
            with st.spinner("Analyzing... The AI is thinking and running tools 🤔"):
                init_msg = [
                    {
                        "role":"user",
                        "content":f"Analyze this customer message by classifying it and analyzing its sentiment: {customer_message}"
                    }
                ]

                init_resp = ollama.chat(
                    model="llama3.2:1b",
                    messages=init_msg,
                    tools=tools,
                )

                print(init_resp['message'])

                msg_for_next_step = init_msg + [init_resp['message']]
                tool_outputs = []

                if "tool_calls" in init_resp['message']:
                    for tool_call in init_resp['message']:
                        name = tool_call['function']['name']
                        args = tool_call["function"]["arguments"]

                        if name == "classify_issue":
                            result = classify_issue(args["message"])
                        elif name == "analyze_sentiment":
                            result = analyze_sentiment(args["message"])
                        else:
                            result = f"Unknown tool call: {name}"

                        tool_outputs.append(
                            {
                                "role":"tool",
                                "name":name,
                                "content":result,
                            }
                        )
                
                if tool_outputs:
                    msg_for_next_step.extend(tool_outputs)
                    final_resp = ollama.chat(
                        model="llama3.2:1b",
                        messages=msg_for_next_step,
                    )

                    resp_text = final_resp['message']['content']
                    actual_resp = re.sub(r"<think>.*?</think>","",resp_text).strip()

                    st.divider()
                    st.subheader("✅ Analysis Complete!")

                    col1, col2 = st.columns(2)

                    # Display the results from the tool calls directly
                    for output in tool_outputs:
                        if output['name'] == 'classify_issue':
                            with col1:
                                st.metric("Issue Category", output['content'])
                        elif output['name'] == 'analyze_sentiment':
                            with col2:
                                st.metric("Sentiment", output['content'])

                    st.subheader("📝 AI-Generated Summary")
                    st.markdown(actual_resp)

                    with st.expander("Show Raw Tool Calls & Model Responses"):
                        st.json({
                            "initial_response_from_model": init_resp,
                            "executed_tool_outputs": tool_outputs,
                            "final_response_from_model": final_resp
                        })

                else:
                    # If no tools were called, show the initial response
                    st.info("The model did not call any specific tools. Here is its direct response:")
                    st.write(init_resp['message']['content'])

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.warning(
                "Please ensure the Ollama application is running and the specified model (e.g., 'qwen2') is installed. You can run `ollama pull qwen2` in your terminal.")

