"""
====================================================================================================
api_local_client.py —— 知识蒸馏模型命令行客户端
====================================================================================================
【快速启动】
    python api_local_client.py
====================================================================================================
"""
import requests
import json
import time

API_URL = "http://127.0.0.1:8892"
DEFAULT_MODEL = 'bilstm'


def check_health():
    """检查服务是否可用"""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def call_predict(text, model_type=DEFAULT_MODEL):
    """调用预测接口"""
    resp = requests.post(
        f"{API_URL}/predict_api",
        json={"text": text, "model_type": model_type},
        timeout=30
    )
    if resp.status_code == 200:
        return resp.json()
    else:
        print(f"错误: {resp.status_code} - {resp.text}")
        return None


def call_model_info(model_type=DEFAULT_MODEL):
    """获取模型信息"""
    resp = requests.get(f"{API_URL}/model_info", params={"model_type": model_type})
    if resp.status_code == 200:
        return resp.json()
    return None


if __name__ == '__main__':
    print("=" * 60)
    print("知识蒸馏模型 - 命令行客户端")
    print("=" * 60)

    if not check_health():
        print("\n❌ 服务未启动！请先运行: python api_fast_service.py")
        exit(1)

    print("✅ 服务连接成功")

    # 获取模型信息
    info = call_model_info()
    if info:
        print(f"\n模型类型: {info['type']}")
        print(f"模型名称: {info['name']}")
        print(f"参数量: {info['params']:,}")
        print(f"类别数: {info['class_cnt']}")
        print(f"设备: {info['device']}")

    print("\n" + "-" * 40)
    print("输入文本进行预测，输入 'q' 退出")
    print("输入 'model=teacher' 切换为教师模型")
    print("输入 'model=bilstm' 切换为 BiLSTM 学生模型")
    print("-" * 40 + "\n")

    current_model = DEFAULT_MODEL

    while True:
        text = input("请输入文本: ").strip()

        if text.lower() == 'q':
            print("退出。")
            break

        if not text:
            continue

        if text.startswith('model='):
            new_model = text.split('=')[1]
            if new_model in ['teacher', 'prune', 'bilstm', 'transformer']:
                current_model = new_model
                print(f"✅ 已切换为: {new_model}\n")
            else:
                print(f"❌ 不支持的模型: {new_model}\n")
            continue

        start = time.perf_counter()
        result = call_predict(text, current_model)
        elapsed = (time.perf_counter() - start) * 1000

        if result:
            print(f"\n  当前模型: {current_model}")
            print(f"  预测类别: {result['class']} (ID: {result['class_id']})")
            print(f"  置信度: {result['confidence']:.4f}")
            print(f"  耗时: {elapsed:.1f} ms")
            print(f"  Top-3 概率:")
            for item in result['probabilities'][:3]:
                bar = '█' * int(item['probability'] * 50)
                print(f"    {item['class']:8s}: {item['probability']:.4f} {bar}")
        print()