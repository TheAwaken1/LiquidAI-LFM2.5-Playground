import gradio as gr
import torch
import torchaudio
import torchaudio.functional as F
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState, LFMModality
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import tempfile
import os
import warnings
import numpy as np
from queue import Queue
from threading import Thread
from fastrtc import AdditionalOutputs, ReplyOnPause, WebRTC

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Audio model
audio_processor = LFM2AudioProcessor.from_pretrained("LiquidAI/LFM2.5-Audio-1.5B")
audio_model = LFM2AudioModel.from_pretrained("LiquidAI/LFM2.5-Audio-1.5B").to(device).eval()

# ADD THESE LINES to match the demo's 'mimi' and 'proc' variables
mimi = audio_processor.mimi  # This is the codec for .streaming()
proc = audio_processor       # This is the processor for text decoding

# VL model
vl_processor = AutoProcessor.from_pretrained("LiquidAI/LFM2.5-VL-1.6B")
vl_model = AutoModelForImageTextToText.from_pretrained(
    "LiquidAI/LFM2.5-VL-1.6B",
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
).eval()

def init_audio_chat_state():
    chat = ChatState(audio_processor)
    chat.new_turn("system")
    chat.add_text("Respond with interleaved text and audio.")
    chat.end_turn()
    return chat

# Theme
theme = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="blue",
    radius_size="md",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
)

with gr.Blocks(theme=theme, title="LiquidAI Multimodal Madness") as demo:
    gr.Markdown("# 💧 LiquidAI Multimodal Madness")

    with gr.Tab("🎙️ Speech-to-Speech Chat"):
        chat_state = gr.State(init_audio_chat_state())

        webrtc = WebRTC(
            modality="audio",
            mode="send-receive",
            full_screen=False,
            # CRITICAL: Disable browser filters that cause crackling
            track_constraints={
                "echoCancellation": False,
                "noiseSuppression": False,
                "autoGainControl": False,
            }
        )

        text_out = gr.Textbox(lines=8, label="Live Response Text", interactive=False)
        clear_btn = gr.Button("Reset Chat")
        export_btn = gr.DownloadButton("📥 Download Transcript", variant="secondary")

        def chat_producer(q: Queue, chat: ChatState, temp: float | None, topk: int | None):
            print(f"Starting generation with state {chat}.")
            with torch.no_grad(), mimi.streaming(1):
                for t in audio_model.generate_interleaved(
                    **chat,
                    max_new_tokens=1024,
                    audio_temperature=temp,
                    audio_top_k=topk,
                ):
                    q.put(t)
                    if t.numel() > 1:
                        if (t == 2048).any():
                            continue
                        # Use mimi to decode exactly like the demo (no .to(device))
                        wav_chunk = mimi.decode(t[None, :, None])[0]
                        q.put(wav_chunk)
            q.put(None)
                
        def chat_response(audio: tuple[int, np.ndarray], _id: str, chat: ChatState, temp: float | None = 1.0, topk: int | None = 4):
            # Match demo parameter handling
            if temp == 0:
                temp = None
            if topk == 0:
                topk = None
            if temp is not None:
                temp = float(temp)
            if topk is not None:
                topk = int(topk)

            if len(chat.text) == 1:
                chat.new_turn("system")
                chat.add_text("Respond with interleaved text and audio.")
                chat.end_turn()
                chat.new_turn("user")

            rate, wav = audio
            chat.add_audio(torch.tensor(wav / 32_768, dtype=torch.float), rate)
            chat.end_turn()

            chat.new_turn("assistant")

            q: Queue = Queue()
            chat_thread = Thread(target=chat_producer, args=(q, chat, temp, topk))
            chat_thread.start()

            out_text: list[torch.Tensor] = []
            out_audio: list[torch.Tensor] = []
            out_modality: list[LFMModality] = []

            while True:
                t = q.get()  # Blocking get - exactly like the demo
                if t is None:
                    break
                elif t.numel() == 1:  # text token
                    out_text.append(t)
                    out_modality.append(LFMModality.TEXT)
                    print(proc.text.decode(t), end="")  # Print to terminal like demo
                    cur_string = proc.text.decode(torch.cat(out_text)).removesuffix("<|text_end|>")
                    yield AdditionalOutputs(cur_string)  # Real-time UI update
                elif t.numel() == 8:  # audio token
                    out_audio.append(t)
                    out_modality.append(LFMModality.AUDIO_OUT)
                elif t.numel() == 1920:  # decoded audio chunk - exact size like demo
                    np_chunk = (t.cpu().numpy() * 32_767).astype(np.int16)
                    yield (24_000, np_chunk)
                else:
                    print(f"Warning: unexpected tensor shape: {t.shape}")

            chat.append(
                text=torch.stack(out_text, 1) if out_text else None,
                audio_out=torch.stack(out_audio, 1) if out_audio else None,
                modality_flag=torch.tensor(out_modality, device="cuda"),
            )
            chat.end_turn()
            chat.new_turn("user")

        webrtc.stream(
            ReplyOnPause(chat_response, input_sample_rate=24_000, output_sample_rate=24_000, can_interrupt=False),
            inputs=[webrtc, chat_state],
            outputs=[webrtc],
        )

        webrtc.on_additional_outputs(lambda s: s, outputs=[text_out])
        
        def export_audio_chat_history(chat: ChatState):
            if not chat or not chat.text: return None
            content = "--- LiquidAI Audio Chat History ---\n\n"
            try:
                for i, turn in enumerate(chat.text):
                    if turn is not None and turn.numel() > 0:
                        role = "System" if i == 0 else ("User" if i % 2 != 0 else "Assistant")
                        text = audio_processor.text.decode(turn.squeeze())
                        content += f"{role}: {text}\n\n"
            except Exception as e: content += f"\n[Error: {str(e)}]"
            fd, path = tempfile.mkstemp(suffix=".txt", text=True)
            with os.fdopen(fd, 'w', encoding='utf-8') as f: f.write(content)
            return path

        export_btn.click(export_audio_chat_history, inputs=[chat_state], outputs=[export_btn])
        clear_btn.click(lambda: (init_audio_chat_state(), ""), outputs=[chat_state, text_out])
        
    # TTS Tab (dropdown restored with voice prompt)
    with gr.Tab("🗣️ Text-to-Speech"):
        gr.Markdown("Pure TTS: Generate speech-only output with selectable voice (via prompt).")
        
        with gr.Row():
            with gr.Column():
                tts_voice = gr.Dropdown(
                    choices=["US male", "US female", "UK male", "UK female"],
                    value="US female",
                    label="Voice"
                )
                tts_text = gr.Textbox(label="Text to speak", lines=8, placeholder="Enter text here...")
                tts_btn = gr.Button("Generate Speech", variant="primary")
            
            with gr.Column():
                tts_output = gr.Audio(
                    label="Generated Speech", 
                    autoplay=True, 
                    waveform_options={"sample_rate": 24000}
                )

        def generate_tts(voice, text):
            if not text.strip():
                return None
            system_prompt = f"Perform TTS. Use the {voice} voice."
            chat = ChatState(audio_processor)
            chat.new_turn("system")
            chat.add_text(system_prompt)
            chat.end_turn()
            chat.new_turn("user")
            chat.add_text(text.strip())
            chat.end_turn()
            chat.new_turn("assistant")
            audio_tokens = []
            for t in audio_model.generate_sequential(**chat, max_new_tokens=1024, audio_temperature=0.8, audio_top_k=64):
                if t.numel() > 1:
                    audio_tokens.append(t)
            if not audio_tokens:
                return None
            codes = torch.stack(audio_tokens[:-1], dim=1).unsqueeze(0).to(device)
            wf = audio_processor.decode(codes).cpu().squeeze(0)
            return (24000, wf.numpy())

        tts_btn.click(generate_tts, inputs=[tts_voice, tts_text], outputs=[tts_output])
    
    # ASR Tab
        # ASR Tab
    with gr.Tab("📝 Speech-to-Text (ASR)"):
        gr.Markdown(
            """
            # Speech-to-Text (ASR)
            Transcribe audio to text. Supports long files.
            """
        )
        with gr.Row():
            with gr.Column():
                asr_mic = gr.Microphone(type="numpy", label="Record Clip")
                asr_upload = gr.Audio(type="filepath", label="Upload Audio File")
                
                with gr.Accordion("Advanced Settings", open=False):
                    asr_max_tokens = gr.Slider(
                        minimum=256, maximum=4096, value=2048, step=256, 
                        label="Max Generation Tokens"
                    )
                
                asr_btn = gr.Button("Transcribe", variant="primary")
            
            with gr.Column():
                asr_output = gr.Textbox(label="Transcription", lines=15)

        def generate_asr(mic_data, upload_path, max_tokens):
            y, audio_sr = None, None
            
            if mic_data is not None:
                sr, data = mic_data
                y = torch.from_numpy(data).float()
                if y.dim() == 1:
                    y = y.unsqueeze(0)
                elif y.dim() == 2:
                    if y.shape[1] > y.shape[0]:
                        y = y.T
                if y.shape[0] > 1:
                    y = y.mean(dim=0, keepdim=True)
                audio_sr = sr
                
            elif upload_path:
                try:
                    import librosa
                    y_np, audio_sr = librosa.load(upload_path, sr=None, mono=True)
                    y = torch.from_numpy(y_np).float().unsqueeze(0)
                except Exception as e:
                    return f"File load error (check file/format?): {str(e)}"
                    
            if y is None:
                return "No audio provided."

            if audio_sr != 24000:
                y = torchaudio.functional.resample(y, orig_freq=audio_sr, new_freq=24000)

            max_amp = y.abs().max()
            if max_amp > 0:
                y = y / max_amp * 0.95

            chat = ChatState(audio_processor)
            chat.new_turn("system")
            chat.add_text("Perform ASR.")
            chat.end_turn()
            chat.new_turn("user")
            chat.add_audio(y.to(device), 24000)
            chat.end_turn()
            chat.new_turn("assistant")
            
            text_token_ids = []
            for t in audio_model.generate_sequential(**chat, max_new_tokens=max_tokens):
                if t.numel() == 1:
                    text_token_ids.append(t.item())
                    
            if not text_token_ids:
                return "No speech detected or transcription empty."
                
            full_text = audio_processor.text.decode(text_token_ids, skip_special_tokens=True)
            return full_text.strip()

        asr_btn.click(generate_asr, inputs=[asr_mic, asr_upload, asr_max_tokens], outputs=[asr_output])

        # Vision-Language Tab (Gradio 6.0 compatible: dict history format, no deprecated params)
    with gr.Tab("🖼️ Vision Chat"):
        gr.Markdown(
            "**Vision-Language** 🔥 Upload multiple images + chat."
        )
        
        # Optional: Wrap chatbot in Column for natural tall height
        with gr.Column():
            vl_chatbot = gr.Chatbot(
                label="Conversation",
                avatar_images=(None, "https://liquid.ai/favicon.ico"),
                type="messages",
                allow_tags=False,
            )
        
        # Bigger gallery with full image fit + click-to-zoom popup
        vl_gallery = gr.Gallery(
            label="Current Images (click any image to zoom full-size)",
            height=700,
            object_fit="contain",
            preview=True,
            allow_preview=True,
            columns=3
        )
        
        with gr.Row():
            vl_images = gr.Files(
                label="Upload Images",
                file_count="multiple",
                file_types=["image"],
                type="filepath"
            )
            vl_prompt = gr.Textbox(label="Question/Prompt", placeholder="Describe? OCR? Reason? Compare these images?", lines=3)
        
        with gr.Row():
            vl_submit = gr.Button("Send", variant="primary")
            vl_clear = gr.Button("Clear Chat")

        vl_history = gr.State([])  # List of dicts: [{"role": "user/assistant", "content": "..."}]

        def vl_chat(prompt, files, history):
            if not prompt and not files:
                return history, history, None

            # Load images (as PIL for gallery display)
            images = []
            gallery_images = []  # For gallery update
            if files:
                for f in files:
                    try:
                        img = Image.open(f).convert("RGB")
                        images.append(img)
                        gallery_images.append(img)
                    except Exception as e:
                        print(f"Error loading image: {e}")

            # Build multimodal content for model input
            conversation = [
                {
                    "role": "user",
                    "content": []
                }
            ]
            
            if images:
                for _ in images:
                    conversation[0]["content"].append({"type": "image"})
            
            user_text = prompt.strip() if prompt else "Describe the image(s)."
            conversation[0]["content"].append({"type": "text", "text": user_text})

            # Prepare inputs using processor
            try:
                text_prompt = vl_processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = vl_processor(text=text_prompt, images=images if images else None, return_tensors="pt")
            except Exception:
                inputs = vl_processor(text=user_text, images=images if images else None, return_tensors="pt")

            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                generated_ids = vl_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7
                )

            # Decode response
            input_len = inputs["input_ids"].shape[1]
            new_tokens = generated_ids[0][input_len:]
            response = vl_processor.decode(new_tokens, skip_special_tokens=True)

            # Build display message (dict format for new Chatbot)
            display_prompt = prompt.strip() if prompt else ("Describe these images." if images else "")
            user_msg = display_prompt + (" (with images)" if images else "")

            # Update history with dicts
            new_history = history + [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": response}
            ]

            # Update gallery
            gallery_update = gallery_images if gallery_images else None

            return new_history, new_history, gallery_update

        vl_submit.click(
            vl_chat,
            inputs=[vl_prompt, vl_images, vl_history],
            outputs=[vl_chatbot, vl_history, vl_gallery]
        )
        vl_clear.click(lambda: ([], [], None), outputs=[vl_chatbot, vl_history, vl_gallery])
        
demo.launch(inbrowser=True)
