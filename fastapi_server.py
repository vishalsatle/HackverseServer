import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from groq_agent_pipeline import (
    build_fallback_brief,
    detect_escalation,
    fetch_disaster_intel,
    fetch_live_osint_news,
    fetch_weather_intel,
    run_intel_pipeline,
    run_monte_carlo_simulation,
    build_source_signal_map,
    evaluate_benchmarks,
)

load_dotenv()


REFRESH_INTERVAL_SECONDS = 5 * 60 


class Article(BaseModel):
    source_id: str
    source_name: str
    title: str
    summary: str
    link: str
    published: str


class Telemetry(BaseModel):
    run_count: int
    success_count: int
    last_duration_ms: int
    last_article_count: int


class BriefResponse(BaseModel):
    live_articles: List[Article]
    brief: str
    generated_at: int          
    running: bool
    next_run_at: int      
    last_error: Optional[str]
    telemetry: Telemetry


class JobState:
    def __init__(self, mode: str, location: str):
        self.key = f"{mode}:{location}"
        self.mode = mode
        self.location = location
        self.data: Optional[Dict] = None
        self.generated_at: int = 0
        self.running: bool = False
        self.last_error: Optional[str] = None
        self.next_run_at: int = int((time.time() + REFRESH_INTERVAL_SECONDS) * 1000)
        self.run_count: int = 0
        self.success_count: int = 0
        self.last_duration_ms: int = 0
        self.last_article_count: int = 0


jobs: Dict[str, JobState] = {}


def _fetch_intel(mode: str, location: str) -> List[Dict]:
    if mode == "weather":
        return fetch_weather_intel(location=location)
    if mode == "disaster":
        return fetch_disaster_intel(top_n=5)
    return fetch_live_osint_news(top_n=5)


def _run_pipeline_sync(mode: str, location: str) -> Dict:
    started = time.perf_counter()
    intel_data = _fetch_intel(mode, location)

    escalation = detect_escalation(intel_data)
    simulations = run_monte_carlo_simulation(escalation=escalation, iterations=400)
    source_signal_map = build_source_signal_map(escalation=escalation, simulations=simulations)
    context_data = {
        "escalation": escalation,
        "simulations": simulations,
        "source_signal_map": source_signal_map,
    }

    try:
        brief = run_intel_pipeline(intel_data, context_data=context_data)
    except Exception as exc:
        brief = (
            build_fallback_brief(intel_data)
            + f"\n\n#### Pipeline Notice\nGroq/CrewAI failed; fallback used. Error: {str(exc)[:300]}"
        )

    elapsed = time.perf_counter() - started
    benchmarks = evaluate_benchmarks(
        elapsed_seconds=elapsed,
        source_signal_map=source_signal_map,
    )

    return {
        "intel_data": intel_data,
        "brief": brief,
        "elapsed_ms": int(elapsed * 1000),
        "benchmarks": benchmarks,
    }


async def execute_job(job: JobState) -> None:
    if job.running:
        return
    job.running = True
    job.run_count += 1
    started_at = time.time()
    try:
        result = await asyncio.to_thread(_run_pipeline_sync, job.mode, job.location)
        job.data = result
        job.generated_at = int(time.time() * 1000)
        job.success_count += 1
        job.last_article_count = len(result["intel_data"])
        job.last_error = None
    except Exception as exc:
        job.last_error = str(exc)[:400]
    finally:
        job.last_duration_ms = int((time.time() - started_at) * 1000)
        job.running = False
        job.next_run_at = int((time.time() + REFRESH_INTERVAL_SECONDS) * 1000)


async def _scheduler_loop() -> None:
    while True:
        await asyncio.sleep(REFRESH_INTERVAL_SECONDS)
        for job in list(jobs.values()):
            asyncio.create_task(execute_job(job))


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_scheduler_loop())
    yield
    task.cancel()

app = FastAPI(
    title="Hackverse Intelligence API",
    description="Multi-source OSINT intelligence pipeline — FastAPI edition",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://hackverse.vishalsatleit24-claude.workers.dev"
    ],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok", "time_utc": datetime.now(timezone.utc).isoformat()}


@app.get("/brief", response_model=BriefResponse)
async def get_brief(
    mode: str = Query(default="conflict", pattern="^(conflict|weather|disaster)$"),
    location: str = Query(default="Kyiv", max_length=80),
    force: bool = Query(default=False),
):
    key = f"{mode}:{location}"

    if key not in jobs:
        jobs[key] = JobState(mode, location)
        asyncio.create_task(execute_job(jobs[key]))

    job = jobs[key]

    if force:
        await execute_job(job)

    if job.data is None:
        raise HTTPException(
            status_code=202,
            detail={
                "error": "Brief abhi generate ho raha hai. Thodi der mein dobara try karo.",
                "running": job.running,
                "next_run_at": job.next_run_at,
                "last_error": job.last_error,
                "telemetry": {
                    "run_count": job.run_count,
                    "success_count": job.success_count,
                    "last_duration_ms": job.last_duration_ms,
                    "last_article_count": job.last_article_count,
                },
            },
        )

    intel_data: List[Dict] = job.data["intel_data"]
    articles = [
        Article(
            source_id=item.get("source_id", "OSINT-UNKNOWN"),
            source_name=item.get("source_name", "Unknown"),
            title=item.get("title", ""),
            summary=item.get("summary", ""),
            link=item.get("link", ""),
            published=item.get("published", ""),
        )
        for item in intel_data
    ]

    return BriefResponse(
        live_articles=articles,
        brief=job.data["brief"],
        generated_at=job.generated_at,
        running=job.running,
        next_run_at=job.next_run_at,
        last_error=job.last_error,
        telemetry=Telemetry(
            run_count=job.run_count,
            success_count=job.success_count,
            last_duration_ms=job.last_duration_ms,
            last_article_count=job.last_article_count,
        ),
    )


@app.get("/jobs")
async def list_jobs():
    return [
        {
            "key": job.key,
            "mode": job.mode,
            "location": job.location,
            "running": job.running,
            "run_count": job.run_count,
            "success_count": job.success_count,
            "last_error": job.last_error,
            "generated_at": job.generated_at,
            "next_run_at": job.next_run_at,
        }
        for job in jobs.values()
    ]
