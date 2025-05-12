import requests
import os

# API 服务器的基地址
BASE_URL = "http://127.0.0.1:8000"  # 或者 http://0.0.0.0:8000，通常 127.0.0.1 更适合客户端调用

# 创建保存音频的目录
SAVE_DIR = "api_test_results"
os.makedirs(SAVE_DIR, exist_ok=True)

def test_create_voice():
    """测试 /tts/create 端点"""
    print("测试 /tts/create 端点...")
    url = f"{BASE_URL}/tts/create"
    data = {
        "text": "你好，这是一个通过API创建的语音。",
        "gender": "female",
        "pitch": 3,
        "speed": 3,
    }
    try:
        response = requests.post(url, data=data)
        if response.status_code == 200:
            # 保存音频文件
            save_path = os.path.join(SAVE_DIR, "created_voice.wav")
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"成功！音频已保存到: {save_path}")
            print(f"响应头 Content-Type: {response.headers.get('Content-Type')}")
        else:
            print(f"错误！状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
    print("-" * 30)

BASE_URL = "http://127.0.0.1:8000"

def test_clone_voice(prompt_audio_path: str):
    """测试 /tts/clone 端点"""
    if not os.path.exists(prompt_audio_path):
        print(f"错误：找不到提示音频文件 '{prompt_audio_path}'。请提供一个有效的 WAV 文件路径。")
        return

    print(f"测试 /tts/clone 端点，使用提示音频: {prompt_audio_path}...")
    url = f"{BASE_URL}/tts/clone"
    data = {
        "text": "这是一个通过API克隆的语音。",
        "prompt_text": "这是提示音频的文本内容，如果适用的话。" # 可选
    }
    files = {
        "prompt_audio": (os.path.basename(prompt_audio_path), open(prompt_audio_path, "rb"), "audio/wav")
    }
    try:
        response = requests.post(url, data=data, files=files)
        if response.status_code == 200:
            # 保存音频文件
            save_path = os.path.join(SAVE_DIR, "cloned_voice.wav")
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"成功！音频已保存到: {save_path}")
            print(f"响应头 Content-Type: {response.headers.get('Content-Type')}")
        else:
            print(f"错误！状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
    finally:
        # 关闭文件句柄
        if 'prompt_audio' in files and files['prompt_audio'][1]:
            files['prompt_audio'][1].close()
    print("-" * 30)

if __name__ == "__main__":
    # 1. 测试语音创建功能
    test_create_voice()

    # 2. 测试语音克隆功能
    #    请将 'path/to/your/prompt_audio.wav' 替换为您本地的一个 WAV 音频文件路径
    #    例如: prompt_audio_file = "example/speaker1_train.wav" (如果您的项目中有这个文件)
    #    或者您可以使用任意一个您想要克隆声音的 WAV 文件。
    prompt_audio_file_path = "/Users/guobingming/projects/voiceclone/test_long_audio.wav" # <--- 修改这里!

    # 为了能直接运行，您可以先创建一个虚拟的 prompt_audio.wav 文件
    # 例如，在与 test_api.py 同目录下创建一个空的 dummy_prompt.wav
    # 或者复制一个您项目中的示例音频到这里
    # 这里我假设您会替换它，或者手动创建一个：
    if prompt_audio_file_path == "path/to/your/prompt_audio.wav":
        print(f"\n请在脚本中将 'prompt_audio_file_path' 修改为实际的WAV文件路径再测试克隆功能。")
        # 您可以取消下面这行的注释，并提供一个实际的文件路径来测试
        # test_clone_voice(prompt_audio_file_path)
    else:
        test_clone_voice(prompt_audio_file_path)

    # 如果您想快速测试并且有 `example/speaker1_train.wav`
    # test_clone_voice("example/speaker1_train.wav")
