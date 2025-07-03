from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import onnxruntime
from numpy.typing import NDArray

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models, onnx_bert_models
from style_bert_vits2.nlp.japanese.g2p import text_to_sep_kata
from style_bert_vits2.utils import get_onnx_device_options


if TYPE_CHECKING:
    import torch


def extract_bert_feature(
    text: str,
    word2ph: list[int],
    device: str,
    assist_text: str | None = None,
    assist_text_weight: float = 0.7,
    sep_text: list[str] | None = None,
) -> torch.Tensor:
    """
    日本語のテキストから BERT の特徴量を抽出する (PyTorch 推論)

    Args:
        text (str): 日本語のテキスト
        word2ph (list[int]): 元のテキストの各文字に音素が何個割り当てられるかを表すリスト
        device (str): 推論に利用するデバイス
        assist_text (str | None, optional): 補助テキスト (デフォルト: None)
        assist_text_weight (float, optional): 補助テキストの重み (デフォルト: 0.7)
        sep_text (list[str] | None, optional): 単語単位の単語のリスト (デフォルト: None)

    Returns:
        torch.Tensor: BERT の特徴量
    """

    import torch

    # sep_text が指定されていない場合のみ、text_to_sep_kata() を使って単語単位の単語のリストを作る
    # 指定されている時はそのリストをそのまま使うことで、何回も形態素解析処理が行われないようにする
    if sep_text is None:
        sep_text = text_to_sep_kata(text, raise_yomi_error=False)[0]

    # 各単語が何文字かを作る `word2ph` を使う必要があるので、読めない文字は必ず無視する
    # でないと `word2ph` の結果とテキストの文字数結果が整合性が取れない
    text = "".join(sep_text)
    if assist_text:
        assist_text = "".join(text_to_sep_kata(assist_text, raise_yomi_error=False)[0])

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = bert_models.load_model(Languages.JP, device_map=device)
    bert_models.transfer_model(Languages.JP, device)

    style_res_mean = None
    with torch.no_grad():
        tokenizer = bert_models.load_tokenizer(Languages.JP)
        if assist_text:
            inputs = tokenizer(assist_text, return_tensors="pt")
        else:
            inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)  # type: ignore
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0]

        res = torch.mean(res, dim=0, keepdim=True).to(res.device)
        total_padding = len(word2ph) - len(res)


        padding = torch.zeros((total_padding, res.shape[1]), dtype=res.dtype, device=res.device)
        res = torch.cat([res , padding], 0).to(res.device)

        linear_transform = torch.nn.Linear(in_features=256, out_features=1024).to(res.device)

        # 変換を実行
        res = linear_transform(res).to(res.device)

    #assert len(word2ph) == len(text) + 2, text
    word2phone = torch.LongTensor(word2ph).to(device)
    if assist_text:
        assert style_res_mean is not None
        repeat_feature = (
            res * (1 - assist_text_weight) + style_res_mean * assist_text_weight
        )
    else:
        repeat_feature = res
    phone_level_feature = torch.repeat_interleave(repeat_feature, word2phone, 0)
    return phone_level_feature.T


def extract_bert_feature_onnx(
    text: str,
    word2ph: list[int],
    onnx_providers: Sequence[str | tuple[str, dict[str, Any]]],
    assist_text: str | None = None,
    assist_text_weight: float = 0.7,
    sep_text: list[str] | None = None,
) -> NDArray[Any]:
    """
    日本語のテキストから BERT の特徴量を抽出する (ONNX 推論)

    Args:
        text (str): 日本語のテキスト
        word2ph (list[int]): 元のテキストの各文字に音素が何個割り当てられるかを表すリスト
        onnx_providers (list[str]): ONNX 推論で利用する ExecutionProvider (CPUExecutionProvider, CUDAExecutionProvider など)
        assist_text (str | None, optional): 補助テキスト (デフォルト: None)
        assist_text_weight (float, optional): 補助テキストの重み (デフォルト: 0.7)
        sep_text (list[str] | None, optional): 単語単位の単語のリスト (デフォルト: None)

    Returns:
        NDArray[Any]: BERT の特徴量
    """

    # sep_text が指定されていない場合のみ、text_to_sep_kata() を使って単語単位の単語のリストを作る
    # 指定されている時はそのリストをそのまま使うことで、何回も形態素解析処理が行われないようにする
    if sep_text is None:
        sep_text = text_to_sep_kata(text, raise_yomi_error=False)[0]

    # 各単語が何文字かを作る `word2ph` を使う必要があるので、読めない文字は必ず無視する
    # でないと `word2ph` の結果とテキストの文字数結果が整合性が取れない
    text = "".join(sep_text)
    if assist_text:
        assist_text = "".join(text_to_sep_kata(assist_text, raise_yomi_error=False)[0])

    # トークナイザーとモデルの読み込み
    tokenizer = onnx_bert_models.load_tokenizer(Languages.JP)
    session = onnx_bert_models.load_model(
        language=Languages.JP,
        onnx_providers=onnx_providers,
    )
    input_names = [input.name for input in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    # 入力テンソルの転送に使用するデバイス種別, デバイス ID, 実行オプションを取得
    device_type, device_id, run_options = get_onnx_device_options(session, onnx_providers)  # fmt: skip

    # 入力をテンソルに変換
    if assist_text:
        inputs = tokenizer(assist_text, return_tensors="np")
    else:
        inputs = tokenizer(text, return_tensors="np")
    input_tensor = [
        inputs["input_ids"].astype(np.int64),  # type: ignore
        inputs["attention_mask"].astype(np.int64),  # type: ignore
    ]
    # 推論デバイスに入力テンソルを割り当て
    ## GPU 推論の場合、device_type + device_id に対応する GPU デバイスに入力テンソルが割り当てられる
    io_binding = session.io_binding()
    for name, value in zip(input_names, input_tensor):
        gpu_tensor = onnxruntime.OrtValue.ortvalue_from_numpy(
            value, device_type, device_id
        )
        io_binding.bind_ortvalue_input(name, gpu_tensor)
    # text から BERT 特徴量を抽出
    io_binding.bind_output(output_name, device_type)
    session.run_with_iobinding(io_binding, run_options=run_options)
    res = io_binding.get_outputs()[0].numpy()

    res =  np.mean(res, axis=0, keepdims=True) # 1行に圧縮


    # --- Linear Transformation ---
    # You'll need the weight and bias from your PyTorch linear_transform.
    # For this example, let's create dummy weight and bias
    in_features = 256
    out_features = 1024
    linear_transform_weight = np.random.rand(out_features, in_features) # PyTorch weight is [out, in]
    linear_transform_bias = np.random.rand(out_features) # PyTorch bias is [out]

    # Perform the linear transformation: Y = X @ W_T + B
    # In NumPy, for Y = XA^T + B, it's X @ A.T + B or X @ A_transposed + B
    # Since PyTorch's linear layer weight is (out_features, in_features),
    # we need to transpose it for the dot product with res (which is [N, in_features]).
    res = res @ linear_transform_weight.T+ linear_transform_bias
    
    res = res.astype(np.float32) 

    total_padding = len(word2ph) - len(res)
    padding = np.zeros((total_padding, res.shape[1]), dtype=res.dtype)
    res = np.concatenate([res, padding], axis=0)

    #assert len(word2ph) == len(text) + 2, text
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = np.tile(res[i], (word2phone[i], 1))
        phone_level_feature.append(repeat_feature)

    phone_level_feature = np.concatenate(phone_level_feature, axis=0)

    return phone_level_feature.T
