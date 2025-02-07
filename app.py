from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import gradio as gr
import spaces

torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = "syubraj/MedicalChat-Phi-3.5-mini-instruct"
try:
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model and Tokenizer loaded successfully.")
except Exception as e:
    raise ValueError(f"Error loading Model and Tokenizer: {e}")

def generate_response(user_query: str, system_message: str = None, max_length: int = 1024) -> str:
    """
    Generates a response based on the given user query.

    :param user_query: The user's input message.
    :param system_message: Custom system instruction (optional, defaults to medical assistant).
    :param max_length: Max tokens to generate.
    :return: Generated assistant response.
    """
    if not user_query.strip():
        return "Error: User query cannot be empty."

    if system_message is None:
        system_message = ("You are a trusted AI-powered medical assistant. "
                          "Analyze patient queries carefully and provide accurate, professional, and empathetic responses. "
                          "Prioritize patient safety, adhere to medical best practices, and recommend consulting a healthcare provider when necessary.")

    messages = [
        {"role": "system", "content": system_message},
        {'role': "user", "content": user_query}
    ]

    pipe = pipeline("text-generation",
                    model=model, 
                    tokenizer=tokenizer)
    
    generation_args = {
        "max_new_tokens": max_length,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
}
    try:
        output = pipe(messages, **generation_args)
        return(output[0]['generated_text'])
    except Exception as e:
        return f"Error generating response: {e}"

# Gradio Interface
def chat_interface(user_query, system_message=None):
    response = generate_response(user_query, system_message)
    return response

with gr.Blocks() as demo:
    gr.Markdown("# Medical Chatbot")
    gr.Markdown("Ask your medical questions, and the AI will provide professional responses.")
    
    with gr.Row():
        user_query = gr.Textbox(label="Your Query", placeholder="Enter your question here...", lines=3)
        system_message = gr.Textbox(label="System Message (Optional)", placeholder="Custom system instruction...", lines=3)
    
    submit_button = gr.Button("Submit")
    output = gr.Textbox(label="Assistant Response", lines=5)
    
    submit_button.click(chat_interface, inputs=[user_query, system_message], outputs=output)

demo.launch()


