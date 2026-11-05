# -*- coding: utf-8 -*-
"""
FastText 新闻分类客户端（命令行）
==================================

功能：通过 HTTP 请求调用 FastAPI 预测服务，支持交互式和批量测试。

快速启动：
    python api_local_client.py
    → 自动检查服务端可用性 → 进入交互式预测模式

运行模式：
    交互模式 — 逐条输入标题，实时返回预测结果
    批量测试 — 预设 5 条测试标题，自动预测并展示结果

依赖：
    - requests：HTTP 客户端库
    - json：JSON 解析（Python 标准库）
    - time：延时控制（Python 标准库）
"""

import requests
import json
import time

# ============================================================
# 全局配置
# ============================================================
# 服务端地址：FastText 服务运行在 8890 端口
BASE_URL = "http://127.0.0.1:8890"

# 预测接口完整 URL
PREDICT_URL = f"{BASE_URL}/predict_api"

# 请求超时时间（秒）
# 设太短：网络波动时误报超时
# 设太长：服务端挂了客户端要等很久才报错
# 5 秒对本地服务来说足够
REQUEST_TIMEOUT = 5


# ============================================================
# 服务端健康检查
# ============================================================
def check_server() -> bool:
    """
    探测服务端是否可用。

    调用 GET /health 接口，成功返回 True，失败返回 False。

    返回：
        bool — True 表示服务端可用，False 表示不可用

    异常处理：
        所有 requests 异常（连接失败、超时等）统一捕获，返回 False。
        不做细粒度区分，因为对调用方来说结果都一样：服务不可用。
    """
    try:
        resp = requests.get(
            f'{BASE_URL}/health',
            timeout=REQUEST_TIMEOUT
        )
        if resp.status_code == 200:
            print(f'[OK] 服务端已就绪：{resp.json()}')
            return True
    except requests.exceptions.RequestException:
        pass
    print(f"[ERROR] 无法连接到服务端 {BASE_URL}，请先启动 api_fast_service.py")
    return False


# ============================================================
# 单条预测
# ============================================================
def predict_title(title: str) -> dict | None:
    """
    向服务端发送单条预测请求。

    参数：
        title : str — 待分类的新闻标题

    返回：
        dict | None — 预测成功返回 {"title": ..., "pred_class": ...}
                      预测失败返回 None

    请求流程：
        1. 构造 JSON payload：{"title": "..."}
        2. POST 到 /predict_api
        3. 检查 HTTP 状态码
        4. 200 → 解析 JSON 返回
        5. 非 200 → 打印错误信息，返回 None

    异常处理（按优先级）：
        ConnectionError — 服务端未启动或端口错误
        Timeout         — 服务端响应超时
        RequestException — 其他网络异常（DNS、SSL 等）
    """
    payload = {"title": title}

    try:
        # requests.post() 参数说明：
        #   url     — 请求地址
        #   json    — 请求体（自动序列化为 JSON + 设置 Content-Type: application/json）
        #   timeout — 超时时间（秒），覆盖全局 REQUEST_TIMEOUT
        #             注意：timeout 是连接 + 读取的总超时，不是单阶段超时
        resp = requests.post(
            PREDICT_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        # HTTP 状态码判断：
        #   200 — 成功，body 中是预测结果 JSON
        #   422 — Pydantic 校验失败（如 title 为空字符串）
        #   500 — 服务端内部错误（如模型文件缺失）
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 422:
            # 422 Unprocessable Entity：请求格式正确但参数不合法
            print(f"[422] 请求参数错误：{resp.json()}")
        else:
            print(f"[{resp.status_code}] 服务端返回错误：{resp.text}")

    except requests.exceptions.ConnectionError:
        # 连接被拒绝：最常见的错误，通常是服务端没启动
        print("[ERROR] 连接被拒绝，请确认服务端已启动。")

    except requests.exceptions.Timeout:
        # 超时：服务端可能卡住了
        print(f"[ERROR] 请求超时（>{REQUEST_TIMEOUT}s）。")

    except requests.exceptions.RequestException as e:
        # 兜底：其他所有请求异常
        print(f"[ERROR] 请求异常：{e}")

    return None


# ============================================================
# 批量测试
# ============================================================
def batch_test():
    """
    批量测试模式：用预设的 5 条标题依次调用预测接口。

    每条标题预测后延时 0.3 秒，防止过快请求。
    如果服务端做了限流，这个延时是必要的。
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
            time.sleep(0.3)  # 礼貌性延时，避免服务端压力过大


# ============================================================
# 交互式模式
# ============================================================
def interactive_mode():
    """
    交互式预测模式：循环读取用户输入，实时调用预测接口。

    操作说明：
        - 输入新闻标题 → 回车 → 查看预测结果
        - 输入 exit     → 退出程序
        - 输入 batch    → 进入批量测试模式
        - Ctrl+C/Ctrl+D → 退出程序

    空输入会被忽略，不会发送请求。
    """
    print("\n" + "=" * 50)
    print("交互式预测模式（输入 exit 退出，输入 batch 批量测试）")
    print("=" * 50)

    while True:
        try:
            title = input("请输入新闻标题：").strip()
        except (EOFError, KeyboardInterrupt):
            # EOFError：Linux/Mac 下 Ctrl+D
            # KeyboardInterrupt：Windows/Linux 下 Ctrl+C
            print("\n已退出。")
            break

        # 空输入 → 跳过，等待下一次输入
        if not title:
            continue

        # 退出命令
        if title.lower() == "exit":
            print("已退出。")
            break

        # 批量测试命令
        if title.lower() == "batch":
            batch_test()
            continue

        # 正常预测
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
        exit(1)  # 非零退出码，表示异常退出

    # 2- 进入交互模式
    interactive_mode()