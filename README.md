# gpt-oss-20b を Unsloth でファインチューニング

2025年8月5日にOpenAIから公開された、オープンウェイトなモデル [gpt-oss](https://openai.com/ja-JP/index/introducing-gpt-oss/) 。
ここでは gpt-oss の軽量な方の gpt-oss-20b を、[Unsloth](https://github.com/unslothai/unsloth) でファインチューニングします。

gpt-oss-20b はローカルでも動きますが、今回は Google Colab の T4 GPU で動作検証をしました。

## ノートブックの構成

1.  **ライブラリのインストール**:
    必要なライブラリをインストールします。

2.  **モデルのロードと設定**:
    Unsloth の `FastLanguageModel` を使用してモデルをロード、パラメータを設定します。

    ```python
    from unsloth import FastLanguageModel
    import torch

    max_seq_length = 1024
    dtype = None

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/gpt-oss-20b",
        dtype = dtype,
        max_seq_length = max_seq_length,
        load_in_4bit = True,
        full_finetuning = False,
    )
    ```

3.  **LoRA アダプターの設定**:
    モデルに LoRA アダプターを追加し、ファインチューニングの対象となるモジュールを指定します。

    ```python
    model = FastLanguageModel.get_peft_model(
        model,
        r = 8,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
    )
    ```

4.  **ファインチューニング前の推論性能を確認**:
    ファインチューニングをする前に、特定のタスクでモデルの性能を確認します。

    ```python
    from transformers import TextStreamer
    # 推論コードの実行
    ```

5.  **データセットの準備と整形**:
    ファインチューニングに使用するデータセットをロードし、モデルの入力形式に合わせてデータを整形します。

    ```python
    from datasets import load_dataset
    from unsloth.chat_templates import standardize_sharegpt
    # データセットのロードと整形コード
    ```

6.  **モデルのファインチューニング**:
    整形されたデータセットを使用して、トレーニングを実行します。`trl` ライブラリの `SFTTrainer` を使用し、必要に応じてパラメータを指定します。

    ```python
    from trl import SFTConfig, SFTTrainer
    # トレーニングコードの実行
    ```

7.  **ファインチューニング後の推論性能を確認**:
    トレーニング完了後、同じタスクでモデルの推論性能を再度確認し、ファインチューニングの効果を評価します。

    ```python
    from transformers import TextStreamer
    # ファインチューニング後の推論コードの実行
    ```
