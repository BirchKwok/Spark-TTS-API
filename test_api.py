import pytest
import os
import uuid
from fastapi.testclient import TestClient
from io import BytesIO


# 导入您的 FastAPI 应用实例
# 假设您的 FastAPI app 实例在 api_server.py 文件中名为 app
try:
    from api_server import app, output_audio_dir as api_output_dir
except ImportError as e:
    print(f"错误：无法从 api_server 导入 app。请确保 api_server.py 在 Python 路径中，并且包含名为 'app' 的 FastAPI 实例。 {e}")
    # 如果无法导入，可以引发异常或设置 app 为 None，让测试跳过
    app = None


# --- 配置 ---
TEST_OUTPUT_DIR = "api_test_results"
# 假设 test.wav 在项目根目录
PROMPT_AUDIO_PATH = "test.wav"

# --- Fixtures ---
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """在测试会话开始时创建测试输出目录"""
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    # 可以在这里添加清理逻辑（如果需要的话），使用 yield

@pytest.fixture(scope="module")
def client():
    """提供一个 TestClient 实例"""
    if app is None:
        pytest.skip("FastAPI app 未能加载，跳过 API 测试")
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope="session")
def prompt_audio_bytes():
    """读取测试音频文件的字节内容"""
    if not os.path.exists(PROMPT_AUDIO_PATH):
        pytest.skip(f"测试音频文件未找到: {PROMPT_AUDIO_PATH}")
    with open(PROMPT_AUDIO_PATH, "rb") as f:
        return f.read()

# --- 测试用例 ---

# === /tts/create Tests ===

def test_create_voice_success(client):
    """测试 /tts/create 成功创建语音"""
    response = client.post(
        "/tts/create",
        data={
            "text": "你好，这是一个通过 TestClient 创建的语音。",
            "gender": "female",
            "pitch": 4,
            "speed": 2,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    # 保存响应内容以供检查（可选）
    save_path = os.path.join(TEST_OUTPUT_DIR, "test_create_success.wav")
    with open(save_path, "wb") as f:
        f.write(response.content)
    print(f"\n[Create Success] 音频保存到: {save_path}")

@pytest.mark.parametrize(
    "param, value, expected_detail",
    [
        ("gender", "unknown", "Invalid gender."),
        ("pitch", 0, "Invalid pitch."),
        ("pitch", 6, "Invalid pitch."),
        ("speed", 0, "Invalid speed."),
        ("speed", 6, "Invalid speed."),
    ],
)
def test_create_voice_invalid_params(client, param, value, expected_detail):
    """测试 /tts/create 无效参数"""
    data = {
        "text": "测试无效参数。",
        "gender": "male",
        "pitch": 3,
        "speed": 3,
        param: value, # 覆盖要测试的无效参数
    }
    response = client.post("/tts/create", data=data)
    assert response.status_code == 400
    assert expected_detail in response.json()["detail"]
    print(f"\n[Create Invalid Param] 测试 {param}={value} -> 状态码 {response.status_code}, 详情: {response.json()['detail']}")


def test_create_voice_idempotency(client):
    """测试 /tts/create 的幂等性"""
    idem_key = str(uuid.uuid4())
    data = {
        "text": "测试幂等性，第一次请求。",
        "gender": "male",
        "pitch": 2,
        "speed": 4,
        "idempotency_key": idem_key,
    }

    # 第一次请求
    response1 = client.post("/tts/create", data=data)
    assert response1.status_code == 200
    assert response1.headers["content-type"] == "audio/wav"
    content1 = response1.content
    save_path1 = os.path.join(TEST_OUTPUT_DIR, f"test_create_idem_{idem_key}_1.wav")
    with open(save_path1, "wb") as f:
        f.write(content1)
    print(f"\n[Create Idempotency] 第一次请求 {idem_key} -> 状态码 {response1.status_code}, 保存到: {save_path1}")


    # 第二次请求（使用相同的 key）
    # 文本可以不同，但因为 key 相同，应该返回缓存的结果
    data["text"] = "测试幂等性，这是第二次请求，文本不同。"
    response2 = client.post("/tts/create", data=data)
    assert response2.status_code == 200
    assert response2.headers["content-type"] == "audio/wav"
    content2 = response2.content
    save_path2 = os.path.join(TEST_OUTPUT_DIR, f"test_create_idem_{idem_key}_2.wav")
    with open(save_path2, "wb") as f:
        f.write(content2)
    print(f"[Create Idempotency] 第二次请求 {idem_key} -> 状态码 {response2.status_code}, 保存到: {save_path2}")


    # 理论上内容应该相同，但在某些情况下 TTS 生成可能存在微小差异
    # 至少检查文件大小是否一致
    assert len(content1) == len(content2)
    # 如果需要严格比较，可以比较字节内容: assert content1 == content2


# === /tts/clone Tests ===

def test_clone_voice_success(client, prompt_audio_bytes):
    """测试 /tts/clone 成功克隆语音（无额外参数）"""
    files = {"prompt_audio": ("test.wav", BytesIO(prompt_audio_bytes), "audio/wav")}
    data = {"text": "这是一个通过 TestClient 克隆的语音。"}

    response = client.post("/tts/clone", data=data, files=files)
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    save_path = os.path.join(TEST_OUTPUT_DIR, "test_clone_success.wav")
    with open(save_path, "wb") as f:
        f.write(response.content)
    print(f"\n[Clone Success] 音频保存到: {save_path}")

def test_clone_voice_with_prompt_text(client, prompt_audio_bytes):
    """测试 /tts/clone 使用 prompt_text"""
    files = {"prompt_audio": ("test.wav", BytesIO(prompt_audio_bytes), "audio/wav")}
    data = {
        "text": "克隆语音，并提供提示文本。",
        "prompt_text": "这是测试音频的文本内容。",
    }
    response = client.post("/tts/clone", data=data, files=files)
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    save_path = os.path.join(TEST_OUTPUT_DIR, "test_clone_with_prompt_text.wav")
    with open(save_path, "wb") as f:
        f.write(response.content)
    print(f"\n[Clone with Prompt Text] 音频保存到: {save_path}")


def test_clone_voice_with_customization(client, prompt_audio_bytes):
    """测试 /tts/clone 使用自定义参数"""
    files = {"prompt_audio": ("test.wav", BytesIO(prompt_audio_bytes), "audio/wav")}
    data = {
        "text": "克隆语音，并自定义性别、音高和语速。",
        "gender": "female",
        "pitch": 1,
        "speed": 5,
    }
    response = client.post("/tts/clone", data=data, files=files)
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    save_path = os.path.join(TEST_OUTPUT_DIR, "test_clone_with_customization.wav")
    with open(save_path, "wb") as f:
        f.write(response.content)
    print(f"\n[Clone with Customization] 音频保存到: {save_path}")


@pytest.mark.parametrize(
    "param, value, expected_detail",
    [
        ("gender", "other", "Invalid gender."),
        ("pitch", -1, "Invalid pitch."),
        ("pitch", 10, "Invalid pitch."),
        ("speed", 0, "Invalid speed."),
        ("speed", 7, "Invalid speed."),
    ],
)
def test_clone_voice_invalid_custom_params(client, prompt_audio_bytes, param, value, expected_detail):
    """测试 /tts/clone 无效的自定义参数"""
    files = {"prompt_audio": ("test.wav", BytesIO(prompt_audio_bytes), "audio/wav")}
    data = {
        "text": "测试克隆时的无效参数。",
        "gender": "male", # 默认值
        "pitch": 3,      # 默认值
        "speed": 3,      # 默认值
        param: value,    # 覆盖要测试的无效参数
    }
    response = client.post("/tts/clone", data=data, files=files)
    assert response.status_code == 400
    assert expected_detail in response.json()["detail"]
    print(f"\n[Clone Invalid Custom Param] 测试 {param}={value} -> 状态码 {response.status_code}, 详情: {response.json()['detail']}")


def test_clone_voice_idempotency(client, prompt_audio_bytes):
    """测试 /tts/clone 的幂等性"""
    idem_key = str(uuid.uuid4())
    files1 = {"prompt_audio": ("test.wav", BytesIO(prompt_audio_bytes), "audio/wav")}
    data1 = {
        "text": "测试克隆幂等性，第一次请求。",
        "prompt_text": "第一次的提示文本",
        "gender": "male",
        "pitch": 5,
        "speed": 1,
        "idempotency_key": idem_key,
    }

    # 第一次请求
    response1 = client.post("/tts/clone", data=data1, files=files1)
    assert response1.status_code == 200
    assert response1.headers["content-type"] == "audio/wav"
    content1 = response1.content
    save_path1 = os.path.join(TEST_OUTPUT_DIR, f"test_clone_idem_{idem_key}_1.wav")
    with open(save_path1, "wb") as f:
        f.write(content1)
    print(f"\n[Clone Idempotency] 第一次请求 {idem_key} -> 状态码 {response1.status_code}, 保存到: {save_path1}")

    # 第二次请求（使用相同的 key，不同的数据和文件模拟）
    # 注意：TestClient 每次发送文件时，文件指针会移动，所以需要重新创建 BytesIO
    files2 = {"prompt_audio": ("test_different_name.wav", BytesIO(prompt_audio_bytes), "audio/wav")}
    data2 = {
        "text": "测试克隆幂等性，这是第二次请求。",
        "prompt_text": "第二次的提示文本", # 不同
        "gender": "female",           # 不同
        "pitch": 2,                   # 不同
        "speed": 4,                   # 不同
        "idempotency_key": idem_key,  # 相同
    }
    response2 = client.post("/tts/clone", data=data2, files=files2)
    assert response2.status_code == 200
    assert response2.headers["content-type"] == "audio/wav"
    content2 = response2.content
    save_path2 = os.path.join(TEST_OUTPUT_DIR, f"test_clone_idem_{idem_key}_2.wav")
    with open(save_path2, "wb") as f:
        f.write(content2)
    print(f"[Clone Idempotency] 第二次请求 {idem_key} -> 状态码 {response2.status_code}, 保存到: {save_path2}")


    # 检查幂等性结果
    assert len(content1) == len(content2)
    # assert content1 == content2 # 同样，可以根据需要启用严格比较

# 可以添加更多测试，例如测试模型未加载时的 503 错误等

# 如果想直接运行此脚本进行测试（需要安装 pytest: pip install pytest）
# if __name__ == "__main__":
#     pytest.main([__file__])
# 通常使用命令行运行: pytest test_api.py
