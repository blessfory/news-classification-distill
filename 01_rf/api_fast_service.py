# -*- coding: utf-8 -*-
"""
FastAPI 新闻分类预测服务
========================

基于 TF-IDF + 随机森林的中文新闻标题分类 API。
将 Flask 版 API 迁移到 FastAPI，提供更高的并发性能和自动文档。

启动方式：
    # 方式一：直接运行本文件
    python api_fast_service.py

    # 方式二：命令行用 uvicorn 启动（支持多进程）
    uvicorn api_fast_service:app --host 127.0.0.1 --port 8888 --reload --workers 4

访问地址：
    - Swagger UI 交互式文档：http://127.0.0.1:8888/docs
    - ReDoc 文档：           http://127.0.0.1:8888/redoc
    - 健康检查：             http://127.0.0.1:8888/health
    - 预测接口（POST）：     http://127.0.0.1:8888/predict_api

【Flask → FastAPI 核心变化】
    Flask                          →  FastAPI
    ──────────────────────────────────────────────────
    app = Flask(__name__)         →  app = FastAPI()
    @app.route(...)                →  @app.get / @app.post(...)
    request.get_json()             →  Pydantic BaseModel 自动解析
    jsonify(result)                →  直接 return dict / Pydantic 模型
    app.run(...)                   →  uvicorn.run(...) 或 uvicorn 命令行
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rf_predict_service import predict

# ============================================================
# 请求体定义
# ============================================================
#
# Pydantic BaseModel 作用：
#   1. 自动从请求体解析 JSON → Python 对象
#   2. 校验类型和必填字段（校验失败自动返回 422，附带详细错误信息）
#   3. 在 Swagger UI 中生成请求示例
#
# Field(..., min_length=1) 表示 title 为必填且不能为空字符串，
# 比单纯的 title: str 校验更严格。


class NewsRequest(BaseModel):
    """新闻分类预测请求体"""

    title: str = Field(..., min_length=1, description="待分类的新闻标题", examples=["同步A股首秀：港股缩量回调"])


class NewsResponse(BaseModel):
    """新闻分类预测响应体"""

    title: str = Field(description="原始新闻标题")
    pred_class: str = Field(description="预测类别名称")


# ============================================================
# FastAPI 应用实例
# ============================================================
#
# 常用参数：
#   - title：Swagger UI 标题
#   - description：API 描述（支持 Markdown）
#   - version：版本号
app = FastAPI(
    title="新闻分类预测服务",
    description="基于 TF-IDF + 随机森林的中文新闻标题分类 API",
    version="1.0.0",
)


# ============================================================
# 路由定义
# ============================================================
#
# 知识点：路由对比（Flask vs FastAPI）
#   Flask:
#       @app.route(rule="/predict_api", methods=["POST"])
#       def predict_api():
#           data = request.get_json()
#           ...
#           return jsonify(result)
#
#   FastAPI:
#       @app.post("/predict_api")
#       def predict_api(request: NewsRequest) -> NewsResponse:
#           ...
#           return result
#
#   区别：
#     1. FastAPI 用 @app.post 替代 @app.route(..., methods=["POST"])
#     2. 请求体通过函数参数类型标注自动解析，不需要 request.get_json()
#     3. 返回 dict/Pydantic 模型即可，不需要 jsonify()
#     4. 支持类型标注 -> Swagger UI 自动生成文档


@app.get(
    "/health",
    tags=["系统"],
    summary="健康检查",
    description="用于负载均衡或监控系统检测服务是否存活",
)
def health_check():
    """健康检查接口，返回服务状态"""
    return {"status": "ok", "service": "news-classifier"}


@app.post(
    "/predict_api",
    response_model=NewsResponse,
    tags=["预测"],
    summary="新闻分类预测",
    description="输入新闻标题，返回预测的分类结果",
)
def predict_api(request: NewsRequest) -> NewsResponse:
    """
    新闻分类预测接口

    - **request**: 包含 title 字段的请求体
    - **返回**: 包含 title 和 pred_class 的响应体

    内部处理流程：
        1. 调用 jieba 对标题分词
        2. 使用 TF-IDF 向量化
        3. 使用随机森林模型预测
        4. 返回类别名称
    """
    try:
        # 将 Pydantic 模型转为 dict，传给 rf_predict_service.predict()
        news_data = {"title": request.title}
        result = predict(news_data)
        return result
    except Exception as e:
        # 捕获预测过程中的异常，返回 500 错误
        raise HTTPException(status_code=500, detail=f"预测失败：{str(e)}")


# ============================================================
# 启动入口
# ============================================================
#
# 知识点：uvicorn.run() 常用参数
#   - "api_fast_service:app"：模块名:FastAPI 实例名
#   - host：监听地址（0.0.0.0 允许外部访问，127.0.0.1 仅本机）
#   - port：端口号
#   - reload：开发模式热重载（代码变动自动重启，生产环境关闭）
#   - workers：工作进程数（生产环境使用，reload=True 时只能为 1）
#   - log_level：日志级别（critical / error / warning / info / debug）
#
# Flask 与 uvicorn 参数对比：
#   Flask          →  FastAPI / uvicorn
#   host="127..."  →  --host 127.0.0.1
#   port=8888      →  --port 8888
#   debug=True     →  --reload
#   -              →  --workers 4（生产环境多进程）

if __name__ == "__main__":
    uvicorn.run(
        "api_fast_service:app",
        host="127.0.0.1",
        port=8888,
        reload=True,
        log_level="info",
    )