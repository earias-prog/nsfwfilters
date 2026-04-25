import os
import gc  # EDIT ADDED: import gc for explicit garbage collection
from io import BytesIO
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import pipeline
import anthropic

app = FastAPI()


# Initialize Claude client
client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

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
    "http://192.168.4.59:8081",
    "https://nsfwfilters-production.up.railway.app",
]
# --------------------------------------------------
# CORS (FIXED VERSION)
# KEYWORD: CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # <-- IMPORTANT: allow all for now
    allow_credentials=False,  # <-- MUST be False when using "*"
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

# Request model
class OBDRequest(BaseModel):
    query: str  # can be "P0303" or full sentence

# Response model (optional but cleaner)
class OBDResponse(BaseModel):
    code: str
    summary: str
    potential_causes: list[str]
    potential_fixes: list[str]
    recommended_parts: list[str]
    disclaimer: str
    
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
        img = None  # EDIT ADDED: initialize img to None so finally block is safe
        try:
            img = Image.open(BytesIO(contents))
            img.load()  # EDIT ADDED: force full decode now so BytesIO can be released
            del contents  # EDIT ADDED: release raw bytes immediately after decode
            classification = classifier(img)
            nsfw_result = next(
                (r for r in classification if r["label"].lower() == "nsfw"),
                None
            )
            del classification  # EDIT ADDED: release classifier output after reading
            if nsfw_result and nsfw_result["score"] >= NSFW_THRESHOLD:
                flagged.append(
                    FlaggedFile(
                        filename=file.filename,
                        reason="NSFW content detected",
                        score=nsfw_result["score"],
                    )
                )
        except HTTPException:
            raise  # EDIT ADDED: let HTTP exceptions propagate normally
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to open image {file.filename}: {exc}",
            )
        finally:
            if img is not None:
                img.close()   # EDIT ADDED: explicitly close PIL image to free pixel buffer
                del img       # EDIT ADDED: drop reference so GC can reclaim memory
            gc.collect()      # EDIT ADDED: force GC after each image to prevent accumulation
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

@app.post("/obd/explain", response_model=OBDResponse)
async def explain_obd(req: OBDRequest):
    try:
        user_input = req.query.strip()

        if not user_input:
            raise HTTPException(status_code=400, detail="Empty query")

        # Strong system-style prompt
        prompt = f"""
You are an experienced automotive mechanic.

A user provided the following OBD code or issue:
"{user_input}"

Your job:
- Explain it simply
- Provide only GENERAL guidance
- DO NOT guarantee fixes
- DO NOT say anything unsafe
- DO NOT give professional or legal advice
- Always assume uncertainty

Return ONLY JSON in this exact format:

{{
  "code": "{user_input}",
  "summary": "short explanation",
  "potential_causes": ["cause1", "cause2"],
  "potential_fixes": ["fix1", "fix2"],
  "recommended_parts": ["part1", "part2"],
  "disclaimer": "This is not professional mechanic advice."
}}

Rules:
- Keep it concise
- Use common car terms
- No markdown
- No extra text outside JSON
"""

        # Call Claude
        response = client.messages.create(
            model="claude-3-haiku-20240307",  # fast + cheap
            max_tokens=400,
            temperature=0.3,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Extract text
        raw_output = response.content[0].text

        # OPTIONAL: you can add JSON parsing here if needed
        # but for MVP we return raw JSON string parsed by frontend or later

        import json
        parsed = json.loads(raw_output)

        return parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
