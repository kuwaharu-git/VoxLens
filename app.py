"""
VoxLens: Audio Transcription and Summarization Application
Streamlit UI for speaker diarization, transcription, and summarization
"""
import streamlit as st
import os
import tempfile
from pathlib import Path
import torch

from diarization import SpeakerDiarizer
from transcription import AudioTranscriber
from summarization import ConversationSummarizer
import config


def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="VoxLens - 音声文字起こし＆要約",
        page_icon="🎙️",
        layout="wide"
    )
    
    # Title and description
    st.title("🎙️ VoxLens - 音声文字起こし＆要約アプリ")
    st.markdown("""
    音声ファイル（MP3/WAV）をアップロードして、話者分離、文字起こし、要約を行います。
    
    **機能:**
    - 🗣️ 話者分離（pyannote.audio）
    - 📝 文字起こし（faster-whisper）
    - 📊 AI要約（LangChain + Ollama）
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # HuggingFace token for pyannote
        hf_token = st.text_input(
            "HuggingFace Token",
            type="password",
            help="pyannote.audioを使用するために必要です"
        )
        
        # Device selection
        use_cuda = st.checkbox(
            "CUDAを使用",
            value=torch.cuda.is_available(),
            disabled=not torch.cuda.is_available()
        )
        
        # MapReduce option
        use_map_reduce = st.checkbox(
            "長い文書にMapReduceを使用",
            value=False,
            help="文字数が多い場合に有効"
        )
        
        st.divider()
        st.markdown("""
        **必要な設定:**
        1. HuggingFace Tokenを入力
        2. Ollamaが起動していることを確認
        3. `llama3.2:8b`モデルをインストール
        
        ```bash
        ollama pull llama3.2:8b
        ```
        """)
    
    # File uploader
    st.header("📁 音声ファイルのアップロード")
    uploaded_file = st.file_uploader(
        "MP3またはWAVファイルを選択",
        type=config.SUPPORTED_FORMATS,
        help="対応フォーマット: MP3, WAV"
    )
    
    if uploaded_file is not None:
        # Display file information
        st.success(f"✅ ファイル: {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)")
        
        # Process button
        if st.button("🚀 処理開始", type="primary"):
            
            # Validate HuggingFace token
            if not hf_token:
                st.error("❌ HuggingFace Tokenを入力してください")
                return
            
            # Save uploaded file to temporary directory
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                audio_path = tmp_file.name
            
            try:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Speaker Diarization
                status_text.text("🗣️ 話者分離を実行中...")
                progress_bar.progress(10)
                
                with st.spinner("話者を分離しています..."):
                    diarizer = SpeakerDiarizer(huggingface_token=hf_token)
                    speaker_segments = diarizer.diarize(audio_path)
                
                st.info(f"検出された話者セグメント数: {len(speaker_segments)}")
                progress_bar.progress(35)
                
                # Step 2: Transcription
                status_text.text("📝 文字起こしを実行中...")
                
                with st.spinner("音声を文字起こししています..."):
                    transcriber = AudioTranscriber()
                    full_transcription = transcriber.transcribe_with_speakers(
                        audio_path,
                        speaker_segments
                    )
                
                progress_bar.progress(70)
                
                # Step 3: Summarization
                status_text.text("📊 要約を生成中...")
                
                with st.spinner("LLMで要約を生成しています..."):
                    summarizer = ConversationSummarizer()
                    summary = summarizer.summarize(
                        full_transcription,
                        use_map_reduce=use_map_reduce
                    )
                
                progress_bar.progress(100)
                status_text.text("✅ 処理完了！")
                
                # Display results
                st.success("🎉 処理が完了しました！")
                
                # Create two columns for results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.header("📄 話者ラベル付き全文")
                    st.text_area(
                        "文字起こし結果",
                        value=full_transcription,
                        height=400,
                        label_visibility="collapsed"
                    )
                    
                    # Download button for transcription
                    st.download_button(
                        label="📥 全文をダウンロード",
                        data=full_transcription,
                        file_name="transcription.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    st.header("📊 要約結果")
                    st.text_area(
                        "要約",
                        value=summary,
                        height=400,
                        label_visibility="collapsed"
                    )
                    
                    # Download button for summary
                    st.download_button(
                        label="📥 要約をダウンロード",
                        data=summary,
                        file_name="summary.txt",
                        mime="text/plain"
                    )
                
            except Exception as e:
                st.error(f"❌ エラーが発生しました: {str(e)}")
                st.exception(e)
            
            finally:
                # Clean up temporary file
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>VoxLens - Powered by pyannote.audio, faster-whisper, and LangChain</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
