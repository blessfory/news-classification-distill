# -*- coding: utf-8 -*-
"""
FastText 新闻分类预测服务（FastAPI）
====================================

功能：将 FastText 模型封装为 RESTful API，提供 HTTP 接口供客户端调用。

快速启动：
    python api_fast_service.py
    → 服务运行在 http://127.0.0.1:8890

API 文档：
    启动后访问 http://127.0.0.1:8890/docs 查看 Swagger UI
    启动后访问 http://127.0.0.1:8890/redoc 查看 ReDoc

接口列表：
    GET  /health       — 健康检查
    POST /predict_api  — 新闻分类预测

与 01_rf、02_xgb 的区别：
    模型加载更快（FastText 是 C++ 原生格式），
    预测速度极快（毫秒级），适合高并发场景。

依赖：
    - fastapi：Web 框架
    - uvicorn：ASGI 服务器
    - pydantic：请求/响应数据校验
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ft_predict_service import predict


# ============================================================
# Pydantic 数据模型：定义请求体和响应体的结构
# ============================================================
# Pydantic 是 FastAPI 的数据校验库，基于 Python 类型注解。
# 自动完成：类型校验、必填检查、文档生成、错误提示。

class NewsRequest(BaseModel):
    """
    新闻分类预测请求体

    Field 参数说明：
        ...              — 必填（等价于 Python 3.10+ 的 Required）
        min_length=1     — 最小长度 1，防止空字符串
        description      — 字段说明，自动显示在 Swagger 文档中
        examples         — 示例值，Swagger 中可直接点击试用
    """
    title: str = Field(
        ...,
        min_length=1,
        description="待分类的新闻标题",
        examples=["同步A股首秀：港股缩量回调"]
    )


class NewsResponse(BaseModel):
    """
    新闻分类预测响应体

    Field 的 description 会出现在 Swagger 文档中，
    对于前端/客户端开发者来说是重要的接口说明。
    """
    title: str = Field(description="原始新闻标题（回显）")
    pred_class: str = Field(description="预测的类别名称，如 '财经'、'科技' 等")


# ============================================================
# FastAPI 应用实例化
# ============================================================
# FastAPI 构造函数参数：
#   title       — API 文档标题（显示在 Swagger 顶部）
#   description — API 文档描述（支持 Markdown）
#   version     — API 版本号
app = FastAPI(
    title="新闻分类预测服务（FastText）",
    description="基于 FastText 的中文新闻标题分类 API，支持 15 个类别",
    version="3.0.0",
)


# ============================================================
# API 端点
# ============================================================

# --- GET /health：健康检查 ---
# 装饰器参数说明：
#   path      — URL 路径，必须以 / 开头
#   tags      — 在 Swagger 中分组显示
#   summary   — 简短描述，显示在 Swagger 列表中
#
# 用途：运维监控（K8s liveness probe）、客户端启动前探测服务是否可用
@app.get('/health', tags=['系统'], summary='健康检查')
def health_check():
    """
    健康检查端点

    返回：
        dict — {"status": "ok", "service": "news-classifier-fasttext"}

    使用场景：
        1. 客户端启动时调用，确认服务可用后再发预测请求
        2. 容器编排（K8s）的 liveness/readiness probe
        3. 监控告警系统定期探测
    """
    return {"status": "ok", "service": "news-classifier-fasttext"}


# --- POST /predict_api：新闻分类预测 ---
# 装饰器参数说明：
#   response_model — 指定响应体的 Pydantic 模型，FastAPI 自动校验和序列化
#   tags           — 在 Swagger 中分组
#   summary        — 简短描述
#
# 函数签名：
#   request: NewsRequest — FastAPI 自动从请求体中解析 JSON 并校验
#   -> NewsResponse      — 返回值自动序列化为 JSON，不符合模型会报错
@app.post('/predict_api', response_model=NewsResponse, tags=['预测'], summary='新闻分类预测')
def predict_api(request: NewsRequest) -> NewsResponse:
    """
    新闻分类预测接口

    请求示例：
        POST /predict_api
        Content-Type: application/json
        {"title": "同步A股首秀：港股缩量回调"}

    响应示例：
        {"title": "同步A股首秀：港股缩量回调", "pred_class": "财经"}

    HTTP 状态码：
        200 — 预测成功
        422 — 请求参数校验失败（如 title 为空）
        500 — 服务器内部错误（如模型文件缺失）
    """
    try:
        news_data = {'title': request.title}
        result = predict(news_data)
        return result
    except Exception as e:
        # HTTPException：FastAPI 内置异常类，自动转为 HTTP 错误响应
        # status_code=500 — 内部服务器错误
        # detail           — 错误详情，返回给客户端
        raise HTTPException(status_code=500, detail=f"预测失败：{str(e)}")


# ============================================================
# 服务启动入口
# ============================================================
# uvicorn.run() 参数说明：
#   app        — 模块路径:应用实例名，格式为 "文件名:变量名"
#                FastAPI 支持三种方式：FastAPI 实例、字符串路径、配置字典
#   host       — 监听地址，127.0.0.1 仅本机访问，0.0.0.0 允许外部访问
#   port       — 监听端口，注意不要与 01_rf(8888)、02_xgb(8889) 冲突
#   reload     — 开发模式：代码变更时自动重启（生产环境应设为 False）
#   log_level  — 日志级别：critical/error/warning/info/debug
if __name__ == '__main__':
    uvicorn.run(
        'api_fast_service:app',
        host='127.0.0.1',
        port=8890,               # FastText 专用端口，避免与 01_rf(8888)、02_xgb(8889) 冲突
        reload=True,             # 仅开发用，生产环境务必关闭
        log_level='info'
    )