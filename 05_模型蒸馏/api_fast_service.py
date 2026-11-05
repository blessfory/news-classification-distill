"""
====================================================================================================
api_fast_service.py —— 知识蒸馏模型预测服务（FastAPI）
====================================================================================================
【快速启动】
    python api_fast_service.py
    → 服务运行在 http://127.0.0.1:8892

【API 文档】
    启动后访问 http://127.0.0.1:8892/docs 查看 Swagger UI

【接口列表】
    GET  /health       — 健康检查
    GET  /model_info   — 模型信息
    POST /predict_api  — 新闻分类预测

【端口规划】
    01_rf（随机森林）     端口 8888
    02_xgb（XGBoost）     端口 8889
    03_fasttext（FastText）端口 8890
    04_bert（BERT）        端口 8891
    06_蒸馏（Distillation）端口 8892 ← 本服务
====================================================================================================
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from predict_service import predict, get_model_info

# ==========================================================================
# FastAPI 应用创建
# ==========================================================================
app = FastAPI(
    title="知识蒸馏模型 - 新闻分类 API",
    description="""
## 知识蒸馏模型预测服务

支持四种模型类型：
- **teacher**：BERT 教师模型（约 110M 参数）
- **prune**：剪枝后 BERT 模型
- **bilstm**：BiLSTM 学生模型（约 9.6M 参数，默认）
- **transformer**：3层 Transformer 学生模型（约 15M 参数）
""",
    version="1.0.0"
)

# 默认模型类型
DEFAULT_MODEL = 'bilstm'


# ==========================================================================
# 请求/响应模型
# ==========================================================================
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="待分类的新闻标题",
        examples=["NBA季后赛湖人队大胜勇士"]
    )
    model_type: str = Field(
        default=DEFAULT_MODEL,
        description="模型类型: teacher / prune / bilstm / transformer",
        examples=["bilstm"]
    )


class ProbabilityItem(BaseModel):
    class_name: str = Field(..., alias='class')
    probability: float


class PredictResponse(BaseModel):
    class_name: str = Field(..., alias='class')
    class_id: int
    confidence: float
    probabilities: list[ProbabilityItem]


class ModelInfoResponse(BaseModel):
    type: str
    name: str
    params: int
    class_cnt: int
    class_list: list[str]
    device: str


# ==========================================================================
# 接口定义
# ==========================================================================
@app.get("/health", tags=["系统"])
def health_check():
    """健康检查"""
    return {"status": "ok", "service": "知识蒸馏模型 - 新闻分类 API"}


@app.get("/model_info", tags=["系统"], response_model=ModelInfoResponse)
def model_info(model_type: str = DEFAULT_MODEL):
    """获取模型信息"""
    try:
        info = get_model_info(model_type)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_api", tags=["预测"], response_model=PredictResponse)
def predict_api(request: PredictRequest):
    """新闻分类预测"""
    try:
        result = predict(request.text, request.model_type)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


# ==========================================================================
# 启动入口
# ==========================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("知识蒸馏模型 - FastAPI 服务")
    print("=" * 60)
    print(f"默认模型: {DEFAULT_MODEL}")
    print(f"API 文档: http://127.0.0.1:8892/docs")
    print(f"健康检查: http://127.0.0.1:8892/health")
    uvicorn.run(app, host="127.0.0.1", port=8892, log_level="info")