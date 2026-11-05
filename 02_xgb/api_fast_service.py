# -*- coding: utf-8 -*-
"""
XGBoost 新闻分类预测 API 服务（FastAPI）
=========================================

基于 Word2Vec + XGBoost 的中文新闻标题分类 API。

启动方式：
    python api_fast_service.py

访问地址：
    - Swagger UI 交互式文档：http://127.0.0.1:8889/docs
    - 健康检查：             http://127.0.0.1:8889/health
    - 预测接口（POST）：     http://127.0.0.1:8889/predict_api
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from xgb_predict_service import predict

# ============================================================
# 请求/响应模型定义
# ============================================================
class NewsRequest(BaseModel):
    """新闻分类预测请求体"""
    title: str = Field(..., min_length=1, description="待分类的新闻标题",
                       examples=["同步A股首秀：港股缩量回调"])


class NewsResponse(BaseModel):
    """新闻分类预测响应体"""
    title: str = Field(description="原始新闻标题")
    pred_class: str = Field(description="预测类别名称")


# ============================================================
# FastAPI 应用实例
# ============================================================
app = FastAPI(
    title="新闻分类预测服务（XGBoost）",
    description="基于 Word2Vec + XGBoost 的中文新闻标题分类 API",
    version="2.0.0",
)


# ============================================================
# 路由定义
# ============================================================
@app.get("/health", tags=["系统"], summary="健康检查")
def health_check():
    """健康检查接口，返回服务状态"""
    return {"status": "ok", "service": "news-classifier-xgb"}


@app.post("/predict_api", response_model=NewsResponse, tags=["预测"], summary="新闻分类预测")
def predict_api(request: NewsRequest) -> NewsResponse:
    """
    新闻分类预测接口

    处理流程：
        1. jieba 分词 → 截取前 30 个词
        2. Word2Vec 查词向量 → 取平均 → 128 维密集向量
        3. XGBoost 预测类别
        4. 类别 ID → 类别名称
    """
    try:
        news_data = {"title": request.title}
        result = predict(news_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败：{str(e)}")


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    uvicorn.run(
        "api_fast_service:app",
        host="127.0.0.1",
        port=8889,          # 与 01_rf 的 8888 区分
        reload=True,
        log_level="info",
    )