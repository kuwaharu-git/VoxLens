# VoxLens クイックスタートガイド

このガイドは、VoxLensを最速で起動するための手順です。

## 🚀 5分でスタート

### ステップ1: 必要なソフトウェアのインストール (5分)

```bash
# Python 3.9以上がインストールされていることを確認
python --version

# Ollamaのインストール (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Windows の場合: https://ollama.ai/ からダウンロード
```

### ステップ2: VoxLensのセットアップ (3分)

```bash
# リポジトリのクローン
git clone https://github.com/kuwaharu-git/VoxLens.git
cd VoxLens

# 依存関係のインストール
pip install -r requirements.txt

# Ollamaモデルのダウンロード
ollama pull llama3.2:8b
```

### ステップ3: HuggingFace トークンの取得 (2分)

1. [HuggingFace](https://huggingface.co/join) でアカウント作成（無料）
2. [トークン生成ページ](https://huggingface.co/settings/tokens)で「New token」をクリック
3. Type: "Read"、Name: "VoxLens" として作成
4. トークンをコピー（後で使用）
5. [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) でライセンスに同意

### ステップ4: 起動 (1分)

```bash
# Ollamaサーバーを起動（別のターミナルで）
ollama serve

# VoxLensを起動（メインターミナルで）
streamlit run app.py
```

ブラウザが自動的に開きます！

### ステップ5: 使ってみる

1. 左サイドバーに先ほどコピーしたHuggingFace Tokenを貼り付け
2. 音声ファイル（MP3またはWAV）をアップロード
3. 「処理開始」ボタンをクリック
4. 完了！結果が表示されます

## 💡 動作確認

インストールが正しくできているか確認：

```bash
python check_installation.py
```

すべて✅になればOKです！

## ⚡ よくある問題と解決方法

### "CUDA not available" と表示される

→ **問題なし！** CPUモードで動作します。処理時間が少し長くなるだけです。

### "Ollama connection failed"

```bash
# 別のターミナルで実行
ollama serve
```

### "pyannote.audio model access denied"

HuggingFaceで[モデルページ](https://huggingface.co/pyannote/speaker-diarization-3.1)のライセンスに同意してください。

## 📊 推奨スペック

| 項目 | 最小 | 推奨 |
|-----|------|------|
| CPU | 2コア | 4コア以上 |
| RAM | 8GB | 16GB以上 |
| GPU | 不要 | CUDA対応GPU (6GB以上) |
| ディスク | 10GB | 20GB以上 |

## 🎯 次のステップ

- [詳細な使用方法](USAGE.md)を読む
- [README](README.md)で機能の詳細を確認
- 実際の音声ファイルで試してみる

## 🆘 サポート

問題が解決しない場合：

1. `check_installation.py`を実行して診断
2. エラーメッセージをコピー
3. [GitHubのIssues](https://github.com/kuwaharu-git/VoxLens/issues)で報告

---

**準備完了！** VoxLensで音声の文字起こしと要約を始めましょう！ 🎉
