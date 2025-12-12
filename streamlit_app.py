import streamlit as st
from openai import OpenAI
from io import BytesIO
try:
    from gtts import gTTS
except Exception:
    gTTS = None
import base64

# Show title and description with avatar next to title
col_title, col_avatar = st.columns([9, 1])
with col_title:
    st.title("뭐든지 물어보게나")
with col_avatar:
    avatar_bytes_top = st.session_state.settings.get("avatar_bytes") if "settings" in st.session_state else None
    avatar_url_top = st.session_state.settings.get("avatar_url") if "settings" in st.session_state else None
    if avatar_bytes_top:
        st.image(avatar_bytes_top, width=192)
    elif avatar_url_top:
        st.image(avatar_url_top, width=192)
    else:
        st.markdown("<div style='font-size:84px;line-height:0.9'>🧞</div>", unsafe_allow_html=True)
st.write(
    "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # TTS helpers
    def generate_tts_audio(text: str):
        if not text:
            return None
        provider = st.session_state.settings.get("tts_provider", "gtts")
        lang = st.session_state.settings.get("tts_language", "ko")
        voice = st.session_state.settings.get("tts_voice", "alloy")
        try:
            if provider == "gtts":
                if gTTS is None:
                    raise RuntimeError("gTTS not installed")
                tts = gTTS(text, lang=lang)
                buf = BytesIO()
                tts.write_to_fp(buf)
                buf.seek(0)
                return buf.read()
            elif provider == "openai":
                try:
                    resp = client.audio.speech.create(
                        model="gpt-4o-mini-tts",
                        voice=voice,
                        input=text,
                    )
                except Exception:
                    resp = client.audio.speech.create(model="gpt-4o-mini-tts", voice=voice, input=text)
                if hasattr(resp, "read"):
                    return resp.read()
                if isinstance(resp, bytes):
                    return resp
                if isinstance(resp, dict):
                    if resp.get("audio"):
                        return resp.get("audio")
                    if resp.get("content"):
                        return resp.get("content")
                if hasattr(resp, "content"):
                    return resp.content
                return None
        except Exception as e:
            st.error(f"TTS provider error: {e}")
            return None

    def play_tts(text: str):
        audio_bytes = generate_tts_audio(text)
        if audio_bytes:
            st.audio(BytesIO(audio_bytes), format="audio/mp3")

    # Initialize default settings in session state
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "model": "gpt-3.5-turbo",
            "system_prompt": "",
            "temperature": 0.7,
            "max_tokens": 512,
            "story_mode": False,
            "story_prompt": (
                "당신은 친절한 지니입니다. 학생이 수학 개념을 물어보면,"
                "모험을 하는 듯한 재밌는 스토리로 아이들에게 개념을 설명하세요."
                "마지막에 한 문장 요약을 제공하세요. 톤은 친근하고 웃긴 말투를 사용하세요."
            ),
            # TTS default settings
            "enable_tts": False,
            "tts_language": "ko",
            "tts_speed": 1.0,
            "tts_read_prompt": False,
            "tts_read_response": True,
            "tts_provider": "gtts",
            "tts_voice": "alloy",
                "tts_read_system_prompt": False,
        }

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar: model settings with expand/collapse (expander)
    with st.sidebar.expander("Model Settings", expanded=True):
        # Model selection
        model_choices = [
            "gpt-3.5-turbo",
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4",
        ]
        default_model = st.session_state.settings.get("model", "gpt-3.5-turbo")
        default_index = model_choices.index(default_model) if default_model in model_choices else 0
        model = st.selectbox("Model", model_choices, index=default_index)

        # System prompt - allows tester to set instructions
        system_prompt = st.text_area(
            "System Prompt (system role)",
            value=st.session_state.settings.get("system_prompt", ""),
            height=120,
        )

        # Storytelling mode toggle and prompt
        story_mode = st.checkbox(
            "Enable Short Comic Storytelling for Math Concepts",
            value=bool(st.session_state.settings.get("story_mode", False)),
        )
        story_prompt = st.text_area(
            "Storytelling Prompt (used when story mode is enabled)",
            value=st.session_state.settings.get("story_prompt", ""),
            height=120,
        )

        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.settings.get("temperature", 0.7)),
            step=0.01,
        )

        # Max tokens slider
        max_tokens = st.slider(
            "Max Tokens",
            min_value=50,
            max_value=4096,
            value=int(st.session_state.settings.get("max_tokens", 512)),
            step=50,
        )

        # Apply settings
        if st.button("Apply Settings"):
            st.session_state.settings["model"] = model
            st.session_state.settings["system_prompt"] = system_prompt
            st.session_state.settings["story_mode"] = bool(story_mode)
            st.session_state.settings["story_prompt"] = story_prompt
            st.session_state.settings["temperature"] = temperature
            st.session_state.settings["max_tokens"] = max_tokens

        # Reset conversation button
        if st.button("Reset Conversation"):
            st.session_state.messages = []
            st.success("Conversation reset.")

        # Avatar: allow uploading image or entering a URL, shown next to title
        st.markdown("---")
        st.write("**Avatar (shown next to title)**")
        avatar_file = st.file_uploader("Upload avatar image", type=["png", "jpg", "jpeg", "gif"])
        avatar_url_input = st.text_input("Or paste an image URL", value=st.session_state.settings.get("avatar_url", ""))
        if st.button("Apply Avatar"):
            if avatar_file:
                # store raw bytes so they persist in session_state
                st.session_state.settings["avatar_bytes"] = avatar_file.getvalue()
                st.session_state.settings["avatar_url"] = ""
            else:
                st.session_state.settings["avatar_url"] = avatar_url_input
                st.session_state.settings["avatar_bytes"] = None
            st.experimental_rerun()

        # TTS Options
        st.markdown("---")
        st.write("**Text to Speech (TTS)**")
        enable_tts = st.checkbox("Enable TTS (reads prompts/responses)", value=bool(st.session_state.settings.get("enable_tts", False)))
        tts_provider = st.selectbox("TTS Provider", ["gtts", "openai"], index=0)
        tts_voice = st.selectbox("TTS Voice", ["alloy", "verse", "sage"], index=0)
        tts_language = st.selectbox("Language", ["ko", "en"], index=0)
        tts_speed = st.slider("Speed (1.0 = normal)", 0.5, 2.0, float(st.session_state.settings.get("tts_speed", 1.0)), step=0.1)
        tts_read_prompt = st.checkbox("Read user prompt aloud", value=bool(st.session_state.settings.get("tts_read_prompt", False)))
        tts_read_response = st.checkbox("Read assistant response aloud", value=bool(st.session_state.settings.get("tts_read_response", True)))
        tts_read_system_prompt = st.checkbox("Read system prompt aloud after Apply TTS Settings", value=bool(st.session_state.settings.get("tts_read_system_prompt", False)))
        if st.button("Apply TTS Settings"):
            st.session_state.settings["enable_tts"] = bool(enable_tts)
            st.session_state.settings["tts_provider"] = tts_provider
            st.session_state.settings["tts_voice"] = tts_voice
            st.session_state.settings["tts_language"] = tts_language
            st.session_state.settings["tts_speed"] = float(tts_speed)
            st.session_state.settings["tts_read_prompt"] = bool(tts_read_prompt)
            st.session_state.settings["tts_read_response"] = bool(tts_read_response)
            st.session_state.settings["tts_read_system_prompt"] = bool(tts_read_system_prompt)
            st.experimental_rerun()
        # Button to immediately read the system prompt
        if st.button("Read System Prompt Now"):
            if st.session_state.settings.get("enable_tts"):
                if gTTS is None:
                    st.warning("gTTS not installed. Please install it in requirements.txt.")
                else:
                    combined = st.session_state.settings.get("system_prompt", "")
                    if st.session_state.settings.get("story_mode") and st.session_state.settings.get("story_prompt"):
                        combined = (combined + "\n\n" + st.session_state.settings.get("story_prompt")) if combined else st.session_state.settings.get("story_prompt")
                    if combined:
                        try:
                            # Use selected provider to generate audio
                            audio_bytes = None
                            sel_provider = st.session_state.settings.get("tts_provider", "gtts")
                            if sel_provider == "gtts":
                                if gTTS is None:
                                    raise RuntimeError("gTTS not installed")
                                tts = gTTS(combined, lang=st.session_state.settings.get("tts_language", "ko"))
                                buf = BytesIO()
                                tts.write_to_fp(buf)
                                buf.seek(0)
                                audio_bytes = buf.read()
                            elif sel_provider == "openai":
                                try:
                                    # Use OpenAI TTS - API may vary by SDK. Attempt to call audio.speech.create
                                    resp = client.audio.speech.create(
                                        model="gpt-4o-mini-tts",
                                        voice=st.session_state.settings.get("tts_voice", "alloy"),
                                        input=combined,
                                    )
                                    # Try to get raw bytes from response
                                    if hasattr(resp, "read"):
                                        audio_bytes = resp.read()
                                    elif isinstance(resp, bytes):
                                        audio_bytes = resp
                                    elif isinstance(resp, dict) and resp.get("audio"):
                                        audio_bytes = resp.get("audio")
                                    elif hasattr(resp, "content"):
                                        audio_bytes = resp.content
                                except Exception:
                                    raise
                            if audio_bytes:
                                st.audio(BytesIO(audio_bytes), format="audio/mp3")
                            buf = BytesIO()
                            tts.write_to_fp(buf)
                            buf.seek(0)
                            st.audio(buf, format="audio/mp3")
                        except Exception as e:
                            st.error(f"TTS failed: {e}")

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field (full width)
    if prompt := st.chat_input("뭐든지 물어보게나"):

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        # TTS: read prompt if enabled
        if st.session_state.settings.get("enable_tts") and st.session_state.settings.get("tts_read_prompt"):
            play_tts(prompt)

        # Make sure settings are up to date from the sidebar
        model = st.session_state.settings.get("model", "gpt-3.5-turbo")
        temperature = float(st.session_state.settings.get("temperature", 0.7))
        max_tokens = int(st.session_state.settings.get("max_tokens", 512))
        system_prompt_text = st.session_state.settings.get("system_prompt", "")
        story_mode_flag = st.session_state.settings.get("story_mode", False)
        story_instruction = st.session_state.settings.get(
            "story_prompt", ""
        )

        # Build messages: ensure system prompt is included as first message if provided
        # Avoid duplicate system messages by filtering any existing system role messages.
        messages_for_api = []
        # Combine system prompt with storytelling instruction only when appropriate.
        combined_system_prompt = system_prompt_text or ""
        # Heuristic: check if the user's prompt asks for a concept explanation
        def is_concept_query(text: str) -> bool:
            if not text:
                return False
            t = text.lower()
            korean_triggers = ["개념", "설명", "알려줘", "가르쳐줘"]
            english_triggers = ["explain", "concept", "what is", "define", "teach me"]
            for kw in korean_triggers + english_triggers:
                if kw in t:
                    return True
            return False
        # Add story instruction to combined system prompt if story_mode is on and prompt is concept-like
        if story_mode_flag and is_concept_query(prompt):
            if combined_system_prompt:
                combined_system_prompt = combined_system_prompt + "\n\n" + story_instruction
            else:
                combined_system_prompt = story_instruction
        # Append system prompt (combined) if present and filter out existing system messages
        if combined_system_prompt:
            messages_for_api.append({"role": "system", "content": combined_system_prompt})
        filtered_messages = [m for m in st.session_state.messages if m.get("role") != "system"]
        messages_for_api.extend([
            {"role": m["role"], "content": m["content"]} for m in filtered_messages
        ])

        # Generate a response using the OpenAI API with user-specified parameters.
        stream = client.chat.completions.create(
            model=model,
            messages=messages_for_api,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
        # TTS: read assistant response if enabled
        if st.session_state.settings.get("enable_tts") and st.session_state.settings.get("tts_read_response"):
            play_tts(response)
