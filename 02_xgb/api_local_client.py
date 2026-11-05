# -*- coding: utf-8 -*-
"""
XGBoost API 本地测试客户端
===========================

用于从命令行测试 FastAPI 预测服务，支持：
    - 服务健康检查
    - 交互式单条预测
    - 批量测试

使用方式：
    python api_local_client.py
"""

import requests
import json

# ============================================================
# 服务地址配置
# ============================================================
BASE_URL = "http://127.0.0.1:8889"
HEALTH_URL = f"{BASE_URL}/health"
PREDICT_URL = f"{BASE_URL}/predict_api"
REQUEST_TIMEOUT = 10  # 请求超时时间（秒）


# ============================================================
# 1- 服务健康检查
# ============================================================
def check_server() -> bool:
    """
    检查 API 服务是否正常运行。

    返回:
        True  — 服务正常
        False — 服务不可用
    """
    print("[INFO] 检查服务状态...")
    try:
        resp = requests.get(HEALTH_URL, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            print(f"[OK] 服务正常：{resp.json()}")
            return True
        else:
            print(f"[WARN] 服务返回异常状态码：{resp.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("[ERROR] 连接被拒绝，请确认服务端已启动：python api_fast_service.py")
        return False
    except requests.exceptions.Timeout:
        print("[ERROR] 请求超时，服务可能负载过高")
        return False


# ============================================================
# 2- 单条预测
# ============================================================
def predict_title(title: str) -> dict | None:
    """
    对单条新闻标题进行预测。

    参数:
        title : str — 新闻标题

    返回:
        dict — {"title": "...", "pred_class": "..."}，失败返回 None
    """
    payload = {"title": title}
    try:
        resp = requests.post(PREDICT_URL, json=payload, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 422:
            print(f"[422] 请求参数错误：{resp.json()}")
        else:
            print(f"[{resp.status_code}] 请求失败：{resp.text}")
    except requests.exceptions.ConnectionError:
        print("[ERROR] 连接被拒绝，请确认服务端已启动。")
    except requests.exceptions.Timeout:
        print("[ERROR] 请求超时。")
    return None


# ============================================================
# 3- 批量测试
# ============================================================
def batch_test(test_cases: list[str]):
    """
    批量测试多条新闻标题。

    参数:
        test_cases : list[str] — 新闻标题列表
    """
    print(f"\n{'='*50}")
    print(f"批量测试（共 {len(test_cases)} 条）")
    print(f"{'='*50}")
    correct = 0
    for i, title in enumerate(test_cases, 1):
        result = predict_title(title)
        if result:
            print(f"  [{i}] {title[:30]:<30s} → {result['pred_class']}")
            correct += 1
        else:
            print(f"  [{i}] {title[:30]:<30s} → 预测失败")
    print(f"\n成功：{correct}/{len(test_cases)}")


# ============================================================
# 4- 交互式预测
# ============================================================
def interactive_mode():
    """交互式预测模式：持续输入标题，输入 exit 退出"""
    print("\n" + "=" * 50)
    print("交互式预测模式（输入 exit 退出）")
    print("=" * 50)
    while True:
        title = input("\n请输入新闻标题：").strip()
        if title.lower() == "exit":
            print("退出交互模式。")
            break
        if not title:
            continue
        result = predict_title(title)
        if result:
            print(f"预测类别：{result['pred_class']}")


# ============================================================
# 主程序入口
# ============================================================
if __name__ == '__main__':
    # 1- 先检查服务是否可用
    if not check_server():
        exit(1)

    # 2- 交互式预测
    interactive_mode()

    # 3- 批量测试示例（取消注释以启用）
    # test_cases = [
    #     "同步A股首秀：港股缩量回调",
    #     "世界杯决赛明日在卡塔尔开战",
    #     "最新研究显示人工智能将改变医疗行业",
    # ]
    # batch_test(test_cases)