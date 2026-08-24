from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.mapper import default_mapper


app = FastAPI(title="美股公司识别 API", version="0.1.0")
mapper = default_mapper()


class IdentifyRequest(BaseModel):
    message: str = Field(min_length=1, description="需要识别的消息")


@app.post("/identify")
def identify(request: IdentifyRequest) -> dict[str, object]:
    matches = mapper.identify(request.message)
    return {
        "status": "matched" if matches else "no_match",
        "companies": [match.to_dict() for match in matches],
    }
