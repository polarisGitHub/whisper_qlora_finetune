from decoder import LoraDecoder

decoder = LoraDecoder("medium", prompt="以下是普通话的句子。")
result = decoder(
    [
        "/home/polaris_he/cv-corpus-12.0-2022-12-07/zh-CN/clips/common_voice_zh-CN_33655758.mp3",
        "/home/polaris_he/cv-corpus-12.0-2022-12-07/zh-CN/clips/common_voice_zh-CN_27937403.mp3",
    ],
    return_timestamps=True,
)
print(result)
