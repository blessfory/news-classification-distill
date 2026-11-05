# -*- coding: utf-8 -*-
"""
FastAPI 新闻分类预测 —— 本地测试客户端
========================================

本脚本与 api_fast_service.py 配套使用，通过 requests 库调用服务端接口。

使用步骤：
    1. 先启动服务端：python api_fast_service.py
       （或 uvicorn api_fast_service:app --host 127.0.0.1 --port 8888 --reload）
    2. 再运行本客户端：python api_local_client.py
    3. 在交互界面输入新闻标题进行分类，输入 exit 退出

调用示例：
    输入：同步A股首秀：港股缩量回调
    输出：{'title': '同步A股首秀：港股缩量回调', 'pred_class': 'finance'}

补充：
    - 也可以直接访问 Swagger UI（http://127.0.0.1:8888/docs）
      在浏览器中以可视化方式测试接口，无需写代码。
"""

import time
import requests

# ============================================================
# 配置区
# ============================================================

# 服务端 API 地址
BASE_URL = "http://127.0.0.1:8888"

# 预测接口完整 URL
PREDICT_URL = f"{BASE_URL}/predict_api"

# 请求超时时间（秒），防止网络卡死
REQUEST_TIMEOUT = 5


def check_server():
    """
    检查服务端是否可用（调用 /health 健康检查接口）

    Returns:
        bool: True 表示服务端正常，False 表示连接失败
    """
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            print(f"[OK] 服务端已就绪：{resp.json()}")
            return True
    except requests.exceptions.RequestException:
        pass
    print(f"[ERROR] 无法连接到服务端 {BASE_URL}，请先启动 api_fast_service.py")
    return False


def predict_title(title):
    """
    调用预测接口，对单条新闻标题进行分类

    Args:
        title (str): 待分类的新闻标题

    Returns:
        dict or None: 预测结果字典；请求失败时返回 None
    """
    payload = {"title": title}

    try:
        # 发送 POST 请求，payload 自动序列化为 JSON
        resp = requests.post(PREDICT_URL, json=payload, timeout=REQUEST_TIMEOUT)

        # 响应状态码检查
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 422:
            # 422 通常是 Pydantic 校验失败（如 title 为空）
            print(f"[422] 请求参数错误：{resp.json()}")
        else:
            print(f"[{resp.status_code}] 服务端返回错误：{resp.text}")

    except requests.exceptions.ConnectionError:
        print("[ERROR] 连接被拒绝，请确认服务端已启动。")
    except requests.exceptions.Timeout:
        print(f"[ERROR] 请求超时（>{REQUEST_TIMEOUT}s）。")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 请求异常：{e}")

    return None


def batch_test():
    """
    批量测试：一次性发送多条新闻标题，观察分类效果

    用于快速验证模型在典型场景下的表现。
    """
    test_titles = [
        "同步A股首秀：港股缩量回调",
        "房地产市场新政出台 多地房价应声下跌",
        "苹果发布新款iPhone 搭载最新芯片",
        "人工智能大模型再创新高 科技股集体涨停",
        "体育总局发布新规 足球联赛即将开战",
    ]

    print("\n" + "=" * 50)
    print("批量测试模式")
    print("=" * 50)
    for title in test_titles:
        result = predict_title(title)
        if result:
            print(f"  输入：{result['title']}")
            print(f"  预测：{result['pred_class']}")
            print("-" * 50)
        time.sleep(0.3)  # 礼貌性延时，避免过快请求


def interactive_mode():
    """
    交互式模式：循环读取用户输入，实时调用预测接口

    输入 'exit' 退出循环；输入 'batch' 进入批量测试模式。
    """
    print("\n" + "=" * 50)
    print("交互式预测模式（输入 exit 退出，输入 batch 批量测试）")
    print("=" * 50)

    while True:
        try:
            title = input("请输入新闻标题：").strip()
        except (EOFError, KeyboardInterrupt):
            # 处理 Ctrl+D / Ctrl+C
            print("\n已退出。")
            break

        if not title:
            continue

        if title.lower() == "exit":
            print("已退出。")
            break

        if title.lower() == "batch":
            batch_test()
            continue

        result = predict_title(title)
        if result:
            print(f"  预测类别：{result['pred_class']}")
            print("-" * 50)


# ============================================================
# 主程序入口
# ============================================================
if __name__ == "__main__":
    # 1- 先检查服务端是否可用
    if not check_server():
        exit(1)

    # 2- 进入交互模式
    interactive_mode()