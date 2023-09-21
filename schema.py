from pydantic import BaseModel


class PredictionRequest(BaseModel):
    model_name: str
    model_version: Optional[str] = None
    text: str
    priority: int
