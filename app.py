import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import time
from datetime import datetime

# Optional: Voice input
try:
    import speech_recognition as sr
    HAS_SPEECH = True
except ImportError:
    HAS_SPEECH = False

# --- Model and Author Info ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
AUTHOR = "HAYTHAM EL-HARRAQ"
GITHUB = "https://github.com/HAYTHAMEL-HARRAQ/tinyllama-chatbot"
HF_LINK = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# --- Prompt Templates ---
PROMPT_TEMPLATES = {
    "Default": "You are a helpful AI assistant.",
    "Coding Tutor": "You are a patient and knowledgeable coding tutor. Explain concepts clearly and give code examples.",
    "Career Advisor": "You are a friendly career advisor. Give practical advice for students and professionals.",
    "Motivational Coach": "You are an enthusiastic motivational coach. Encourage and inspire the user.",
    "Joke Bot": "You are a witty AI that loves to tell jokes and puns."
}

# --- Load model and tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", trust_remote_code=True)
model.eval()
tokenizer.pad_token = tokenizer.eos_token

# --- Helper functions ---
def build_prompt(user_input, history, system_prompt):
    prompt = f"{system_prompt}\n\n"
    for turn in history:
        prompt += f"User: {turn['user']}\nAssistant: {turn['bot']}\n"
    prompt += f"User: {user_input}\nAssistant:"
    return prompt

def save_chat_to_file(history, filename="chat_history.json"):
    with open(filename, "w") as f:
        json.dump(history, f, indent=2)

def load_chat_from_file(file):
    if file is None:
        return []
    with open(file.name, "r") as f:
        data = json.load(f)
    # Convert to Gradio format
    return [(turn["user"], turn["bot"]) for turn in data]

def generate_response(user_input, chat_history, system_prompt):
    prompt = build_prompt(user_input, chat_history, system_prompt)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    output_ids = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + 100,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    bot_response = output_text[len(prompt):].strip().split("\n")[0]
    chat_history.append({"user": user_input, "bot": bot_response})
    return bot_response

def gradio_chat(user_input, history, system_prompt, start_time, stats):
    if history is None:
        history = []
    internal = [{"user": x[0], "bot": x[1]} for x in history]
    bot_reply = generate_response(user_input, internal, system_prompt)
    history.append((user_input, bot_reply))
    # Save chat
    save_chat_to_file([{"user": u, "bot": b} for u, b in history])
    # Update stats
    stats["messages"] = len(history) * 2
    stats["words"] = sum(len(u.split()) + len(b.split()) for u, b in history)
    stats["duration"] = int(time.time() - start_time)
    return history, history, "", stats

def transcribe(audio_file):
    if not HAS_SPEECH or audio_file is None:
        return "[Speech recognition not available]"
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        return "[Error: Could not recognize speech]"

def voice_to_text(audio, history, system_prompt, start_time, stats):
    text = transcribe(audio)
    if text:
        return gradio_chat(text, history, system_prompt, start_time, stats)
    return history, history, "", stats

def clear():
    return [], [], "", {"messages": 0, "words": 0, "duration": 0}

def download_history(history):
    # Save as downloadable JSON
    data = [{"user": u, "bot": b} for u, b in history]
    return json.dumps(data, indent=2)

def show_about():
    return f"""
**TinyLlama Chatbot**  
Model: [{MODEL_NAME}]({HF_LINK})  
Author: [{AUTHOR}](https://github.com/HAYTHAMEL-HARRAQ)  
[GitHub Repo]({GITHUB})

Built with [Transformers](https://huggingface.co/docs/transformers/index), [Gradio](https://gradio.app/), and [TinyLlama](https://huggingface.co/TinyLlama).
"""

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 🦙 TinyLlama Chatbot")
    gr.Markdown("A privacy-friendly, local LLM chatbot with extra features. [GitHub Repo](https://github.com/HAYTHAMEL-HARRAQ/tinyllama-chatbot)")

    with gr.Row():
        system_prompt = gr.Dropdown(
            choices=list(PROMPT_TEMPLATES.keys()),
            value="Default",
            label="System Prompt (Role)",
            info="Choose the assistant's personality"
        )
        about_btn = gr.Button("ℹ️ About", elem_id="about-btn")
        about_modal = gr.Markdown(visible=False)

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Type your message", placeholder="Say something...", lines=1)
    if HAS_SPEECH:
        audio = gr.Audio(type="filepath", label="🎙️ Speak to TinyLlama", interactive=True)
    state = gr.State([])
    stats = gr.State({"messages": 0, "words": 0, "duration": 0})
    start_time = gr.State(int(time.time()))
    clear_btn = gr.Button("🧹 Clear Chat")
    download_btn = gr.Button("⬇️ Download Chat History")
    upload_file = gr.File(label="⬆️ Upload Chat History (JSON)", file_types=[".json"])

    with gr.Row():
        msg.submit(
            gradio_chat,
            [msg, state, system_prompt, start_time, stats],
            [chatbot, state, msg, stats]
        )
        if HAS_SPEECH:
            audio.change(
                voice_to_text,
                [audio, state, system_prompt, start_time, stats],
                [chatbot, state, msg, stats]
            )
        clear_btn.click(clear, outputs=[chatbot, state, msg, stats])
        download_btn.click(
            download_history,
            inputs=[state],
            outputs=gr.File(label="Download chat_history.json")
        )
        upload_file.change(
            lambda f: (load_chat_from_file(f), load_chat_from_file(f), "", {"messages": 0, "words": 0, "duration": 0}),
            inputs=[upload_file],
            outputs=[chatbot, state, msg, stats]
        )

    with gr.Row():
        gr.Markdown("### 📊 Session Analytics")
        stats_box = gr.JSON(label="Stats", value={"messages": 0, "words": 0, "duration": 0})

    def update_stats(stats):
        return stats

    stats.change(update_stats, inputs=stats, outputs=stats_box)

    def show_about_modal():
        return gr.update(visible=True, value=show_about())

    about_btn.click(show_about_modal, outputs=about_modal)

demo.launch()