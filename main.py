import asyncio
import heapq
import uuid

from fastapi import FastAPI, HTTPException
from ray import serve
from schemas import PredictionRequest

from model import SentimentAnalysisModel

app = FastAPI()

tasks = []
results = {}

model_versions = {
    "textattack/bert-base-uncased-imdb": None,
    "nlptown/bert-base-multilingual-uncased-sentiment": None,
}


@app.on_event("startup")
async def startup_event():
    ray.init()
    serve.start()
    for model_name, versions in model_versions.items():
        for version in versions:
            model_version = version if version else ""
            model = SentimentAnalysisModel(model_name, model_version)
            serve.create_backend(f"{model_name}_{model_version}", model)
            serve.create_endpoint(
                f"{model_name}_{model_version}",
                backend=f"{model_name}_{model_version}",
                route=f"/{model_name}/{model_version}",
            )
    asyncio.create_task(process_tasks())


async def process_tasks():
    while True:
        if tasks:
            try:
                priority, task_id, task = heapq.heappop(tasks)
                results[task_id] = await task
            except Exception as e:
                results[task_id] = HTTPException(status_code=500, detail=str(e))
        await asyncio.sleep(0.01)


@app.post("/predict")
async def predict(request: PredictionRequest):
    model_name = request.model_name
    model_version = request.model_version
    text = request.text
    priority = request.priority
    if model_name not in model_versions:
        raise HTTPException(status_code=400, detail="Invalid model name")
    if model_version not in model_versions[model_name]:
        raise HTTPException(status_code=400, detail="Invalid model version")
    if not isinstance(priority, int) or priority < 0:
        raise HTTPException(status_code=400, detail="Invalid priority level")
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    task_id = str(uuid.uuid4())
    task = (
        -priority,
        task_id,
        serve.get_handle(f"{model_name}_{model_version}").remote(text),
    )
    heapq.heappush(tasks, task)
    return {"task_id": task_id}


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    if task_id not in results:
        raise HTTPException(status_code=404, detail="Task not found")
    result = results[task_id]
    if isinstance(result, ray.ObjectRef):
        if ray.wait([result], timeout=0)[0]:
            results[task_id] = ray.get(result)
        else:
            return {"status": "pending"}
    return {"status": "completed", "result": results[task_id]}
