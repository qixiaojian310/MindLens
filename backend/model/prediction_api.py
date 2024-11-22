from typing import Union, List

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Feature(BaseModel):
    name: str
    value: bool


class Request(BaseModel):
    features: List[Feature]


class Result(BaseModel):
    BipolarDisorder: bool
    Depression: bool
    AnxietyDisorder: bool
    Schizophrenia: bool
    PTSD: bool


@app.post("/prediction")
def getPrediction(req: Request):
    print(req)

    # TODO 模型预测

    res = Result(
        BipolarDisorder=True,
        Depression=False,
        AnxietyDisorder=False,
        Schizophrenia=False,
        PTSD=False,
    )
    return {"result": res}
