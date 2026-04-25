import os
import json, re
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
    allow_origins=["*"],      # <-- IMPORTANT: allow all for now
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


# Response model kept in file for reference/backward compatibility.
# NOTE: The /obd/explain route no longer uses response_model=OBDResponse
# because Claude may return arrays in slightly different shapes, so we normalize manually.
class OBDResponse(BaseModel):
    code: str
    summary: str
    potential_causes: list[str]
    potential_fixes: list[str]
    recommended_parts: list[str]
    disclaimer: str


# --------------------------------------------------
# OBD RESPONSE NORMALIZER
# KEYWORD: OBD_NORMALIZER
# --------------------------------------------------
def _flatten_to_strings(value) -> list[str]:
    """
    Coerce Claude's array fields into a flat list[str].

    Handles:
      - ["a", "b"]                        -> ["a", "b"]
      - [{"cause": "a"}, {"cause": "b"}]  -> ["a", "b"]
      - "a, b, c"                         -> ["a, b, c"]

    Anything else becomes its string representation so the API does not crash.
    """
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if not isinstance(value, list):
        return [str(value)]

    out: list[str] = []

    for item in value:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            picked = next(
                (v for v in item.values() if isinstance(v, str) and v.strip()),
                None
            )
            out.append(picked if picked else json.dumps(item, ensure_ascii=False))
        else:
            out.append(str(item))

    return out


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


# --------------------------------------------------
# OBD EXPLAIN ROUTE
# KEYWORD: OBD_EXPLAIN
# --------------------------------------------------
@app.post("/obd/explain")
async def explain_obd(req: OBDRequest):
    try:
        user_input = req.query.strip()

        if not user_input:
            raise HTTPException(status_code=400, detail="Empty query")

        prompt = f"""
You are an experienced automotive mechanic.

User input may contain multiple OBD codes and/or symptoms:
\"\"\"{user_input}\"\"\"

Your job:
- Explain the codes and symptoms simply
- Provide only GENERAL guidance
- DO NOT guarantee fixes
- DO NOT say anything unsafe
- DO NOT give professional or legal advice
- Always assume uncertainty
- If multiple codes are present, treat them as a related set and explain any common root cause

Return ONLY JSON in this exact format.
Every array element MUST be a flat string, never an object:

{{
  "code": "<comma-separated P-codes if any, otherwise a 1-line summary of symptoms>",
  "summary": "short combined explanation covering all codes/symptoms",
  "potential_causes": ["cause1", "cause2", "cause3"],
  "potential_fixes": ["fix1", "fix2", "fix3"],
  "recommended_parts": ["part1", "part2"],
  "disclaimer": "This is not professional mechanic advice."
}}

Rules:
- Keep each array element under 120 characters
- Use common car terms
- No markdown
- No code fences
- No extra text outside JSON
"""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )

        raw_output = response.content[0].text.strip()

        # Keep your existing JSON cleanup logic.
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_output)

        if fence_match:
            raw_output = fence_match.group(1).strip()
        else:
            start = raw_output.find("{")
            end = raw_output.rfind("}")

            if start != -1 and end != -1:
                raw_output = raw_output[start: end + 1]

        parsed = json.loads(raw_output)

        # Normalize Claude's output shape so the frontend always receives
        # clean flat string arrays.
        normalized = {
            "code": str(parsed.get("code") or user_input),
            "summary": str(parsed.get("summary") or "").strip(),
            "potential_causes": _flatten_to_strings(parsed.get("potential_causes")),
            "potential_fixes": _flatten_to_strings(parsed.get("potential_fixes")),
            "recommended_parts": _flatten_to_strings(parsed.get("recommended_parts")),
            "disclaimer": str(
                parsed.get("disclaimer")
                or "This is not professional mechanic advice."
            ),
        }

        return normalized

    except HTTPException:
        raise

    except json.JSONDecodeError as e:
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=502,
            detail=f"Claude returned invalid JSON: {str(e)[:200]}"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {str(e)}"
        )
