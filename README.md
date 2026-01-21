# VoxLens 🎙️

音声ファイルの話者分離、文字起こし、要約を行うStreamlitアプリケーション

> 📖 **すぐに始めたい方**: [クイックスタートガイド](QUICKSTART.md)をご覧ください

## 概要

VoxLensは、音声ファイル（MP3/WAV）をアップロードして、以下の処理を自動的に行うアプリケーションです：

1. **話者分離**: pyannote.audio（speaker-diarization-3.1）で話者を識別し、発話区間（タイムスタンプ）を取得
2. **文字起こし**: faster-whisper（distil-large-v3）でCUDA実行し、各区間を文字起こし
3. **要約生成**: LangChain（LCEL）とOllama（llama3.2:8b）で話者関係性を考慮した要約を作成

## 機能

- 🗣️ **話者分離**: 複数の話者を自動的に識別
- 📝 **高精度文字起こし**: GPU加速による高速処理
- 📊 **AI要約**: 話者間の関係性を考慮したインテリジェントな要約
- 💾 **結果のダウンロード**: 全文と要約の両方をテキストファイルとして保存可能
- 🎨 **直感的なUI**: Streamlitによる使いやすいインターフェース

## 必要な環境

- Python 3.9以上
- CUDA対応GPU（推奨、CPUでも動作可能）
- Ollama（ローカルLLMサーバー）
- HuggingFace アカウントとアクセストークン

## インストール

### 1. リポジトリのクローン

```bash
git clone https://github.com/kuwaharu-git/VoxLens.git
cd VoxLens
```

### 2. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 3. Ollamaのインストールと設定

Ollamaをインストールし、必要なモデルをダウンロードします：

```bash
# Ollamaのインストール（詳細は https://ollama.ai/ を参照）
curl -fsSL https://ollama.ai/install.sh | sh

# llama3.2モデルのダウンロード
ollama pull llama3.2:8b
```

### 4. HuggingFace トークンの取得

pyannote.audioモデルを使用するには、HuggingFaceのアクセストークンが必要です：

1. [HuggingFace](https://huggingface.co/)でアカウントを作成
2. [Settings → Access Tokens](https://huggingface.co/settings/tokens)でトークンを生成
3. [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)のライセンスに同意

## 使い方

### 1. Ollamaサーバーの起動

```bash
ollama serve
```

### 2. Streamlitアプリケーションの起動

```bash
streamlit run app.py
```

### 3. ブラウザでアプリケーションを使用

1. ブラウザで `http://localhost:8501` にアクセス
2. サイドバーにHuggingFace Tokenを入力
3. 音声ファイル（MP3またはWAV）をアップロード
4. 「処理開始」ボタンをクリック
5. 処理完了後、以下の結果が表示されます：
   - **左側**: 話者ラベル付き全文（`SPEAKER_XX: 発言内容`形式）
   - **右側**: AI生成の要約

## プロジェクト構成

```
VoxLens/
├── app.py                 # Streamlitメインアプリケーション
├── config.py              # 設定ファイル（モデルサイズの変更はここで行います）
├── diarization.py         # 話者分離モジュール
├── transcription.py       # 文字起こしモジュール
├── summarization.py       # 要約モジュール
├── requirements.txt       # 依存関係
└── README.md             # このファイル
```

## カスタマイズ

### モデルサイズの変更

`config.py`でモデルサイズを変更できます：

**文字起こしモデル（faster-whisper）:**
- 利用可能なサイズ: `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`, `distil-large-v2`, `distil-large-v3`
- 小さいモデルは高速ですが精度が低く、大きいモデルは高精度ですが処理が遅くなります

**要約モデル（Ollama LLM）:**
- 利用可能なサイズ: `llama3.2:1b`, `llama3.2:3b`, `llama3.2:8b`, `llama3.1:8b`, `llama3.1:70b`
- モデルを変更した場合は `ollama pull <モデル名>` でダウンロードが必要です

詳細は[USAGE.md](USAGE.md)を参照してください。

## 技術スタック

- **UI**: Streamlit
- **話者分離**: pyannote.audio (speaker-diarization-3.1)
- **文字起こし**: faster-whisper (distil-large-v3)
- **要約**: LangChain + Ollama (llama3.2:8b)
- **GPU加速**: CUDA/PyTorch

## 設定のカスタマイズ

`config.py`ファイルで以下の設定を変更できます：

- モデル名
- デバイス設定（CUDA/CPU）
- Ollama APIのURL
- サポートする音声フォーマット

## トラブルシューティング

### CUDAエラー

GPUが使用できない場合、`config.py`で`DEVICE = "cpu"`に変更してください。

### Ollamaの接続エラー

Ollamaサーバーが起動していることを確認してください：

```bash
ollama serve
```

### HuggingFaceトークンエラー

- トークンが正しいことを確認
- pyannote.audioのモデルライセンスに同意していることを確認

## ライセンス

このプロジェクトのライセンスについては、LICENSEファイルを参照してください。

## 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずissueを開いて変更内容を議論してください。
