import os
from io import BytesIO
from typing import List, Optional

from pydantic import BaseModel

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
from transformers import pipeline

app = FastAPI()

# --------------------------------------------------
# CORS
# KEYWORD: CORS
# Allow your Expo web dev server + localhost variants
# --------------------------------------------------
allowed_origins = [
    "http://localhost:8081",
    "http://127.0.0.1:8081",
    "http://localhost:19006",
    "http://127.0.0.1:19006",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# MODEL
# KEYWORD: MODEL
# --------------------------------------------------
classifier = pipeline(
    "image-classification",
    model="Falconsai/nsfw_image_detection"
)

NSFW_THRESHOLD = 0.5


class FlaggedFile(BaseModel):
    filename: Optional[str] = None
    reason: Optional[str] = None
    score: Optional[float] = None


class ModerationResponse(BaseModel):
    blocked: bool
    flagged_files: Optional[List[FlaggedFile]] = None
    message: Optional[str] = None


@app.get("/")
def root():
    return {"ok": True, "message": "NSFW moderation API is running."}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/upload-images", response_model=ModerationResponse)
async def upload_images(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No images uploaded.")

    accepted_types = {"image/jpeg", "image/png", "image/gif", "image/webp"}
    flagged: List[FlaggedFile] = []

    for file in files:
        if file.content_type not in accepted_types:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file type: {file.content_type}. "
                    f"Accepted types: {', '.join(sorted(accepted_types))}."
                ),
            )

        contents = await file.read()

        try:
            img = Image.open(BytesIO(contents))
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to open image {file.filename}: {exc}",
            )

        classification = classifier(img)
        nsfw_result = next(
            (r for r in classification if r["label"].lower() == "nsfw"),
            None
        )

        if nsfw_result and nsfw_result["score"] >= NSFW_THRESHOLD:
            flagged.append(
                FlaggedFile(
                    filename=file.filename,
                    reason="NSFW content detected",
                    score=nsfw_result["score"],
                )
            )

    blocked = len(flagged) > 0

    return ModerationResponse(
        blocked=blocked,
        flagged_files=flagged if flagged else None,
        message=(
            "One or more images were blocked due to NSFW content."
            if blocked
            else "All images passed moderation."
        ),
    )
