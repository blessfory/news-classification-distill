"""
====================================================================================================
api_local_client.py —— BERT 新闻分类客户端（命令行）
====================================================================================================
【文件作用】
    通过 HTTP 请求调用 BERT FastAPI 预测服务，支持交互式预测和批量测试。

【快速启动】
    python api_local_client.py
    → 自动检查服务端可用性 → 进入交互式预测模式

【运行模式】
    - 交互模式：逐条输入标题，实时返回预测结果
    - 批量测试：输入 batch 命令，用预设标题自动测试

【依赖包详细说明】
----------------------------------------------------------------------------------------------------
1. requests
   Python 最流行的 HTTP 客户端库，简洁易用。
   常用 API：
   - requests.get(url, timeout)    发送 GET 请求
   - requests.post(url, json, timeout)  发送 POST 请求（JSON 格式）
       参数：
         url:     请求地址
         json:    请求体（dict，自动序列化为 JSON 并设置 Content-Type）
         data:    请求体（dict/str，以表单形式发送）
         params:  URL 查询参数（dict）
         headers: 自定义请求头（dict）
         timeout: 超时时间（秒），None 表示无限等待
                  推荐始终设置 timeout，防止服务端无响应时客户端永久阻塞
       返回：Response 对象
         - .status_code:  HTTP 状态码（200=成功, 422=参数错误, 500=服务器错误）
         - .json():       解析 JSON 响应体为 dict
         - .text:         响应体原始字符串

   异常处理（按继承层级从具体到一般）：
   - requests.exceptions.ConnectionError:  连接被拒绝（服务端未启动）
   - requests.exceptions.Timeout:          请求超时
   - requests.exceptions.HTTPError:        HTTP 错误状态码
   - requests.exceptions.RequestException: 所有异常基类（兜底）

2. json
   Python 标准库，JSON 编码/解码。
   常用 API：
   - json.dumps(obj, indent, ensure_ascii)  Python 对象 → JSON 字符串
       参数：
         indent:        缩进空格数（美化输出）
         ensure_ascii:  True=非 ASCII 转义为 \\u（默认），False=保留中文
   - json.loads(str)                        JSON 字符串 → Python 对象

3. time
   Python 标准库，时间相关操作。
   常用 API：
   - time.sleep(seconds)  暂停指定秒数

【面试高频考点 ⭐】
    1. requests 库的 GET 和 POST 请求怎么写？
    2. requests.post 的 json 参数做了什么？（自动序列化 + 设置 Content-Type）
    3. HTTP 状态码 200/422/500 的含义？如何处理不同状态码？
    4. requests 的异常处理层级？（ConnectionError → Timeout → RequestException）
    5. 为什么要设置 timeout？（防止服务端无响应时客户端永久阻塞）
    6. json.dumps 的 ensure_ascii=False 参数作用？（保留中文不转义）
    7. exit(0) 和 exit(1) 的区别？（正常退出 vs 异常退出）
====================================================================================================
"""
import requests
import json
import time

# ==========================================================================
# 全局配置
# ==========================================================================
# 服务端地址：BERT 服务运行在 8891 端口
# 端口分配：01_rf=8888, 02_xgb=8889, 03_fasttext=8890, 04_bert=8891
BASE_URL = "http://127.0.0.1:8891"

# 预测接口完整 URL
PREDICT_URL = f"{BASE_URL}/predict_api"

# 请求超时时间（秒）
# BERT 模型预测较慢（CPU 上可能 100~500ms），适当放宽超时
# 设太短：正常请求也超时
# 设太长：服务端挂了客户端要等很久才报错
REQUEST_TIMEOUT = 15


# ==========================================================================
# 服务端健康检查
# ==========================================================================
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


# ==========================================================================
# 单条预测
# ==========================================================================
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
        4. 200 OK → 解析 JSON 返回
        5. 非 200 → 打印错误信息，返回 None

    异常处理（按优先级从具体到一般）：
        ConnectionError — 服务端未启动或端口错误
        Timeout         — 服务端响应超时（BERT 模型加载或预测耗时过长）
        RequestException — 其他网络异常（DNS、SSL 等）
    """
    payload = {"title": title}

    try:
        # requests.post() 参数说明：
        #   url     — 请求地址
        #   json    — 请求体，自动序列化为 JSON + 设置 Content-Type: application/json
        #   timeout — 超时时间（秒），覆盖全局默认值
        #             注意：timeout 是连接 + 读取的总超时
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
        # 超时：BERT 模型在 CPU 上可能较慢，或服务端卡住了
        print(f"[ERROR] 请求超时（>{REQUEST_TIMEOUT}s），BERT 模型在 CPU 上可能较慢。")

    except requests.exceptions.RequestException as e:
        # 兜底：其他所有请求异常
        print(f"[ERROR] 请求异常：{e}")

    return None


# ==========================================================================
# 批量测试
# ==========================================================================
def batch_test():
    """
    批量测试模式：用预设的多条标题依次调用预测接口。

    每条标题预测后延时 0.5 秒，防止过快请求。
    BERT 预测比传统模型慢，延时稍长。
    """
    test_titles = [
        "近期新盘推荐 通州纯新别墅本周开盘",
        "房地产市场新政出台 多地房价应声下跌",
        "苹果发布新款iPhone 搭载最新芯片",
        "人工智能大模型再创新高 科技股集体涨停",
        "体育总局发布新规 足球联赛即将开战",
        "NBA季后赛 湖人队逆转取胜",
        "教育部发布双减政策最新进展",
        "新能源汽车销量持续增长",
    ]

    print("\n" + "=" * 50)
    print("批量测试模式（BERT）")
    print("=" * 50)

    for idx, title in enumerate(test_titles, start=1):
        result = predict_title(title)

        if result:
            print(f"  [{idx}] 输入：{result['title']}")
            print(f"       预测：{result['pred_class']}")
            print("-" * 50)
            time.sleep(0.5)  # 礼貌性延时，避免服务端压力过大
        else:
            print(f"  [{idx}] 预测失败：{title[:30]}...")
            print("-" * 50)


# ==========================================================================
# 交互式模式
# ==========================================================================
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
    print("交互式预测模式（BERT）")
    print("输入 exit 退出，输入 batch 批量测试")
    print("=" * 50)

    while True:
        try:
            title = input("\n请输入新闻标题：").strip()
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


# ==========================================================================
# 主程序入口
# ==========================================================================
if __name__ == "__main__":
    # 1- 先检查服务端是否可用
    if not check_server():
        exit(1)  # 非零退出码，表示异常退出

    # 2- 进入交互模式
    interactive_mode()