from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.downloader import download_data
from app.services.script import run_pipeline

router = APIRouter()


class AnalysisRequest(BaseModel):
    latitude: float
    longitude: float
    start_date: str
    end_date: str


@router.post("/analysis")
def run_analysis(data: AnalysisRequest):
    try:

        download_result = download_data(
            latitude=data.latitude,
            longitude=data.longitude,
            start_date=data.start_date,
            end_date=data.end_date
        )

        processing_result = run_pipeline()

        return {
            "download": download_result,
            "analysis": processing_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))