# VoxLens使用ガイド

## 初回セットアップ

### 1. システム要件の確認

- Python 3.9以上
- pip（Pythonパッケージマネージャー）
- GPU（推奨）: CUDA対応のNVIDIA GPU
  - CPU環境でも動作しますが、処理時間が長くなります

### 2. 依存関係のインストール

```bash
# 仮想環境の作成（推奨）
python -m venv venv

# 仮想環境の有効化
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 依存関係のインストール
pip install -r requirements.txt
```

### 3. Ollamaのセットアップ

#### Ollamaのインストール

**macOS/Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
[Ollama公式サイト](https://ollama.ai/)からインストーラーをダウンロード

#### モデルのダウンロード

```bash
ollama pull llama3.2:8b
```

#### Ollamaサーバーの起動

```bash
ollama serve
```

別のターミナルウィンドウで実行してください。サーバーは `http://localhost:11434` で起動します。

### 4. HuggingFace認証の設定

1. [HuggingFace](https://huggingface.co/)にアカウント作成
2. [Access Tokens](https://huggingface.co/settings/tokens)ページでトークンを生成
   - Token type: "Read"で十分
3. [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)モデルページでライセンスに同意

## アプリケーションの起動

```bash
streamlit run app.py
```

ブラウザが自動的に開き、`http://localhost:8501`にアクセスします。

## 使用方法

### 基本的なワークフロー

1. **サイドバーで設定**
   - HuggingFace Tokenを入力
   - CUDA使用の有無を選択（GPUがある場合）
   - 長い文書の場合、MapReduceオプションを有効化

2. **音声ファイルのアップロード**
   - 対応フォーマット: MP3, WAV
   - ファイルサイズ: 推奨200MB以下

3. **処理の実行**
   - 「処理開始」ボタンをクリック
   - 進捗バーで処理状況を確認

4. **結果の確認**
   - 左側: 話者ラベル付き全文
   - 右側: AI生成の要約

5. **結果のダウンロード**
   - 各セクションの下部にあるダウンロードボタンで保存

## 処理時間の目安

音声の長さによって処理時間が変わります：

| 音声の長さ | GPU（CUDA） | CPU |
|---------|-----------|-----|
| 5分 | 約2-3分 | 約10-15分 |
| 15分 | 約5-8分 | 約30-40分 |
| 30分 | 約10-15分 | 約60-90分 |

## トラブルシューティング

### エラー: "HuggingFace Tokenを入力してください"

- サイドバーでHuggingFace Tokenが入力されているか確認
- トークンが有効か確認（HuggingFaceでトークンを再生成）

### エラー: "pyannote.audioモデルにアクセスできません"

- HuggingFaceでモデルのライセンスに同意しているか確認
- トークンに適切な権限があるか確認

### エラー: "Ollamaに接続できません"

```bash
# Ollamaサーバーが起動しているか確認
curl http://localhost:11434

# モデルがインストールされているか確認
ollama list

# llama3.2:8bがない場合はインストール
ollama pull llama3.2:8b
```

### GPU/CUDAエラー

```bash
# CUDAが利用可能か確認
python -c "import torch; print(torch.cuda.is_available())"

# CUDAが使えない場合、config.pyを編集
# DEVICE = "cpu" に変更
```

### メモリ不足エラー

- より小さい音声ファイルで試す
- CPUモードで実行する
- MapReduceオプションを有効化する

## 高度な使用方法

### カスタムモデルの使用

`config.py`を編集して、異なるモデルサイズを使用できます：

```python
# 文字起こしモデルのサイズを変更
# 利用可能なオプション: tiny, base, small, medium, large, large-v2, large-v3, distil-large-v2, distil-large-v3
# 小さいモデル（処理速度優先）: tiny, base, small
# 大きいモデル（精度優先）: large-v3, distil-large-v3
TRANSCRIPTION_MODEL = "large-v3"

# 要約モデルのサイズを変更
# 利用可能なオプション: llama3.2:1b, llama3.2:3b, llama3.2:8b, llama3.1:8b, llama3.1:70b
# 小さいモデル（処理速度優先）: llama3.2:1b, llama3.2:3b
# 大きいモデル（品質優先）: llama3.2:8b, llama3.1:70b
LLM_MODEL = "llama3.2:3b"
```

**注意**: モデルを変更した場合は、Ollamaで事前にダウンロードする必要があります：

```bash
# 例: llama3.2:3bをダウンロード
ollama pull llama3.2:3b
```

### バッチ処理

複数のファイルを処理する場合は、以下のようなスクリプトを作成できます：

```python
from diarization import SpeakerDiarizer
from transcription import AudioTranscriber
from summarization import ConversationSummarizer

# 初期化
diarizer = SpeakerDiarizer(huggingface_token="YOUR_TOKEN")
transcriber = AudioTranscriber()
summarizer = ConversationSummarizer()

# ファイルリスト
files = ["audio1.mp3", "audio2.wav"]

for audio_file in files:
    # 処理
    segments = diarizer.diarize(audio_file)
    transcription = transcriber.transcribe_with_speakers(audio_file, segments)
    summary = summarizer.summarize(transcription)
    
    # 保存
    with open(f"{audio_file}_result.txt", "w") as f:
        f.write(f"Transcription:\n{transcription}\n\nSummary:\n{summary}")
```

## よくある質問（FAQ）

**Q: どの言語に対応していますか？**
A: 現在、日本語に最適化されていますが、`transcription.py`の`language`パラメータを変更することで他の言語にも対応できます。

**Q: オフラインで使用できますか？**
A: モデルを事前にダウンロードしておけば、インターネット接続なしで使用できます。

**Q: 商用利用は可能ですか？**
A: 各モデルのライセンスを確認してください。pyannote.audioとWhisperは商用利用が可能ですが、llama3.2のライセンスも確認が必要です。

## パフォーマンスの最適化

### GPU使用の最適化

```python
# config.pyで設定
COMPUTE_TYPE = "float16"  # より高速
# または
COMPUTE_TYPE = "int8"     # メモリ節約
```

### 並列処理

複数のGPUがある場合、環境変数で指定：

```bash
CUDA_VISIBLE_DEVICES=0,1 streamlit run app.py
```

## サポート

問題が発生した場合は、GitHubのIssuesで報告してください。
