"""
====================================================================================================
api_fast_service.py —— BERT 新闻分类预测服务（FastAPI）
====================================================================================================
【文件作用】
    将 BERT 模型封装为 RESTful API，提供 HTTP 接口供客户端调用。

【快速启动】
    python api_fast_service.py
    → 服务运行在 http://127.0.0.1:8891

【API 文档】
    启动后访问 http://127.0.0.1:8891/docs 查看 Swagger UI（交互式 API 文档）
    启动后访问 http://127.0.0.1:8891/redoc 查看 ReDoc（只读 API 文档）

【接口列表】
    GET  /health       — 健康检查
    POST /predict_api  — 新闻分类预测

【与其他模型的区别】
    01_rf（随机森林）    端口 8888
    02_xgb（XGBoost）    端口 8889
    03_fasttext（FastText）端口 8890
    04_bert（BERT）       端口 8891 ← 本服务

    BERT 预测相对较慢（需 GPU 加速），单次预测约 10~50ms（GPU）/ 100~500ms（CPU）

【依赖包详细说明】
----------------------------------------------------------------------------------------------------
1. fastapi
   现代高性能 Python Web 框架，基于 Starlette 和 Pydantic。
   特点：
   - 自动生成 OpenAPI/Swagger 文档
   - 基于 Python 类型注解的请求/响应校验
   - 异步支持（async/await）
   - 性能接近 Node.js 和 Go

   常用 API：
   - FastAPI(title, description, version)  创建应用实例
       参数：
         title:       API 文档标题
         description: API 文档描述（支持 Markdown）
         version:     版本号

   - @app.get(path, tags, summary)          GET 请求装饰器
   - @app.post(path, response_model, tags)  POST 请求装饰器
       参数：
         path:            URL 路径
         response_model:  响应体的 Pydantic 模型（自动校验和序列化）
         tags:             Swagger 分组标签
         summary:          简短描述

   - HTTPException(status_code, detail)     抛出 HTTP 异常
       参数：
         status_code: HTTP 状态码（400/422/500等）
         detail:      错误详情信息

2. pydantic
   数据校验库，基于 Python 类型注解。
   常用 API：
   - BaseModel:              所有数据模型的基类
   - Field(default, ...)     定义字段属性
       参数：
         default:     默认值
         ...:         必填（Ellipsis），等价于 Python 3.10+ 的 Required
         min_length:  最小长度
         max_length:  最大长度
         description: 字段描述（显示在 Swagger 文档中）
         examples:    示例值列表

3. uvicorn
   ASGI 服务器，用于运行 FastAPI 应用。
   常用 API：
   - uvicorn.run(app, host, port, reload, log_level)
       参数：
         app:        FastAPI 应用实例 或 "模块:变量名" 字符串
         host:       监听地址，'127.0.0.1'（仅本机）/'0.0.0.0'（外部可访问）
         port:       监听端口
         reload:     True=开发模式（代码变更自动重启），False=生产模式
         log_level:  日志级别（critical/error/warning/info/debug）

【面试高频考点 ⭐】
    1. FastAPI 相比 Flask/Django 的优势？（自动生成文档、类型校验、异步支持、高性能）
    2. Pydantic BaseModel 的作用？（数据校验、序列化、自动生成 Swagger 文档）
    3. GET 和 POST 的区别？什么场景用哪个？
    4. HTTP 状态码 200/400/422/500 的含义？
    5. 健康检查接口（/health）的作用？（K8s probe、监控告警、客户端探测）
    6. 127.0.0.1 和 0.0.0.0 的区别？（仅本机 vs 外部可访问）
    7. uvicorn 的 reload 参数作用？（开发模式自动重启，生产环境必须关闭）
    8. RESTful API 的设计原则？
    9. FastAPI 的 async/await 异步支持是如何实现的？（基于 Starlette/ASGI）
====================================================================================================
"""
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from bert_predict_service import predict


# ==========================================================================
# Pydantic 数据模型：定义请求体和响应体的结构
# ==========================================================================
# Pydantic 是 FastAPI 的数据校验库，基于 Python 类型注解。
# 自动完成：类型校验、必填检查、字段文档生成、错误提示。

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
        examples=["近期新盘推荐 通州纯新别墅本周开盘"]
    )


class NewsResponse(BaseModel):
    """
    新闻分类预测响应体

    Field 的 description 会出现在 Swagger 文档中，
    对于前端/客户端开发者来说是重要的接口说明。
    """
    title: str = Field(description="原始新闻标题（回显）")
    pred_class: str = Field(description="预测的类别名称")


# ==========================================================================
# FastAPI 应用实例化
# ==========================================================================
# FastAPI 构造函数参数：
#   title       — API 文档标题（显示在 Swagger 顶部）
#   description — API 文档描述（支持 Markdown）
#   version     — API 版本号
app = FastAPI(
    title="新闻分类预测服务（BERT）",
    description="基于 BERT 预训练模型的中文新闻标题分类 API",
    version="4.0.0",
)


# ==========================================================================
# API 端点
# ==========================================================================

# ------------------------------------------------------------------
# GET /health：健康检查端点
# ------------------------------------------------------------------
# 装饰器参数说明：
#   path      — URL 路径，必须以 / 开头
#   tags      — 在 Swagger 文档中分组显示
#   summary   — 简短描述，显示在 Swagger 列表中
#
# 用途：
#   1. 运维监控（K8s liveness/readiness probe）
#   2. 客户端启动前探测服务是否可用
#   3. 监控告警系统定期探测
@app.get('/health', tags=['系统'], summary='健康检查')
def health_check():
    """
    健康检查端点

    返回：
        dict — {"status": "ok", "service": "news-classifier-bert"}

    使用场景：
        - 客户端启动时调用，确认服务可用后再发预测请求
        - 容器编排（K8s）的 liveness/readiness probe
        - 监控告警系统定期探测
    """
    return {"status": "ok", "service": "news-classifier-bert"}


# ------------------------------------------------------------------
# POST /predict_api：新闻分类预测
# ------------------------------------------------------------------
# 装饰器参数说明：
#   response_model — 指定响应体的 Pydantic 模型，FastAPI 自动校验和序列化输出
#   tags           — 在 Swagger 文档中分组
#   summary        — 简短描述
#
# 函数签名说明：
#   request: NewsRequest — FastAPI 自动从请求体 JSON 解析并校验
#   -> NewsResponse      — 返回值自动序列化为 JSON，不符合模型定义会报错
@app.post('/predict_api', response_model=NewsResponse, tags=['预测'], summary='新闻分类预测')
def predict_api(request: NewsRequest) -> NewsResponse:
    """
    新闻分类预测接口

    请求示例：
        POST /predict_api
        Content-Type: application/json
        {"title": "近期新盘推荐 通州纯新别墅本周开盘"}

    响应示例：
        {"title": "近期新盘推荐 通州纯新别墅本周开盘", "pred_class": "房产"}

    HTTP 状态码：
        200 — 预测成功
        422 — 请求参数校验失败（如 title 为空）
        500 — 服务器内部错误（如模型文件缺失）
    """
    try:
        # 构造预测服务要求的输入格式
        news_data = {'title': request.title}
        # 调用预测函数，返回 {"title": "...", "pred_class": "..."}
        result = predict(news_data)
        return result
    except Exception as e:
        # HTTPException：FastAPI 内置异常类，自动转为 HTTP 错误响应
        # status_code=500 — 内部服务器错误
        # detail           — 错误详情，返回给客户端
        raise HTTPException(status_code=500, detail=f"预测失败：{str(e)}")


# ==========================================================================
# 服务启动入口
# ==========================================================================
# uvicorn.run() 参数说明：
#   app        — 字符串格式 "文件名:应用实例变量名"
#                FastAPI 支持三种方式：FastAPI 实例、字符串路径、配置字典
#   host       — 监听地址，127.0.0.1（仅本机访问），0.0.0.0（允许外部访问）
#   port       — 监听端口，BERT 使用 8891
#               端口分配：01_rf=8888, 02_xgb=8889, 03_fasttext=8890, 04_bert=8891
#   reload     — 开发模式：代码变更时自动重启（生产环境应设为 False）
#   log_level  — 日志级别：critical/error/warning/info/debug
if __name__ == '__main__':
    uvicorn.run(
        'api_fast_service:app',
        host='127.0.0.1',
        port=8891,               # BERT 专用端口，避免与其他模型冲突
        reload=True,             # 仅开发用，生产环境务必关闭
        log_level='info'
    )