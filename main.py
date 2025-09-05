import streamlit as st
from pytube import YouTube
from haystack.nodes import PromptNode, PromptModel
from haystack.nodes.audio import WhisperTranscriber
from haystack.pipelines import Pipeline
from model_add import LlamaCPPInvocationLayer
import time
import os

st.set_page_config(layout="wide")

# ----------------------------
# Download video audio
# ----------------------------
def download_video(url):
    yt = YouTube(url)
    video = yt.streams.filter(abr='160kbps').last()
    return video.download()

# ----------------------------
# Initialize model
# ----------------------------
def initialize_model(full_path):
    return PromptModel(
        model_name_or_path=full_path,
        invocation_layer_class=LlamaCPPInvocationLayer,
        use_gpu=False,
        max_length=512
    )

# ----------------------------
# Initialize prompt node
# ----------------------------
def initialize_prompt_node(model):
    summary_prompt = "deepset/summarization"
    return PromptNode(model_name_or_path=model, 
                      default_prompt_template=summary_prompt, 
                      use_gpu=False)

# ----------------------------
# Transcribe + summarize
# ----------------------------
def transcribe_audio(file_path, prompt_node):
    whisper = WhisperTranscriber()
    pipeline = Pipeline()
    pipeline.add_node(component=whisper, name="whisper", inputs=["File"])
    pipeline.add_node(component=prompt_node, name="prompt", inputs=["whisper"])
    output = pipeline.run(file_paths=[file_path])
    return output

# ----------------------------
# Main App
# ----------------------------
def main():
    st.title("YouTube Video Summarizer 🎥")
    st.markdown(
        '<style>h1{color: orange; text-align: center;}</style>', 
        unsafe_allow_html=True
    )
    st.subheader("Built with Llama 2 🦙, Haystack, Streamlit and ❤️")

    # Expander for details
    with st.expander("About the App"):
        st.write("This app allows you to summarize a YouTube video.")
        st.write("Enter a YouTube URL in the input box below and click 'Submit'.")

    # Input box
    youtube_url = st.text_input("Enter YouTube URL")

    # Submit button
    if st.button("Submit") and youtube_url:
        try:
            start_time = time.time()
            status = st.empty()
            status.info("📥 Downloading video...")

            # Download audio
            file_path = download_video(youtube_url)

            status.info("⚙️ Loading model...")
            full_path = "llama-2-7b-32k-instruct.Q4_K_S.gguf"
            if not os.path.exists(full_path):
                st.error(f"Model file not found: {full_path}")
                return

            model = initialize_model(full_path)
            prompt_node = initialize_prompt_node(model)

            status.info("📝 Transcribing and summarizing...")
            output = transcribe_audio(file_path, prompt_node)

            # Layout
            col1, col2 = st.columns([1, 1])

            with col1:
                st.video(youtube_url)

            with col2:
                st.header("Summarization Result")
                st.write(output)
                try:
                    st.success(output["results"][0].split("\n\n[INST]")[0])
                except Exception:
                    st.warning("Could not parse summary output")

                end_time = time.time()
                st.write(f"⏱ Time taken: {end_time - start_time:.2f} seconds")

            status.success("✅ Done!")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
