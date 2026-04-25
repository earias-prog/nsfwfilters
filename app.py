import os
import json, re
import gc
from io import BytesIO
from typing import List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

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
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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


class OBDRequest(BaseModel):
    query: str


class PlateLookupRequest(BaseModel):
    plate: str
    state: str


class OBDResponse(BaseModel):
    code: str
    summary: str
    potential_causes: list[str]
    potential_fixes: list[str]
    recommended_parts: list[str]
    disclaimer: str


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def _flatten_to_strings(value) -> list[str]:
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


def _first_dict(value):
    if isinstance(value, dict):
        return value

    if isinstance(value, list):
        return next((item for item in value if isinstance(item, dict)), {})

    return {}


def _pick_vehicle_payload(data: dict):
    vehicle = (
        data.get("data")
        or data.get("vehicle")
        or data.get("attributes")
        or data.get("result")
        or data.get("results")
        or data
    )

    return _first_dict(vehicle)


def _get_first(vehicle: dict, *keys):
    for key in keys:
        value = vehicle.get(key)
        if value is not None and str(value).strip():
            return value
    return None


@app.get("/")
def root():
    return {"ok": True, "message": "DiBuy backend API is running."}


@app.get("/health")
def health():
    return {"ok": True}


# --------------------------------------------------
# LICENSE PLATE DECODER
# KEYWORD: PLATE_DECODER
# --------------------------------------------------
@app.post("/decode-license-plate")
def decode_license_plate(req: PlateLookupRequest):
    plate = re.sub(r"[^A-Z0-9]", "", req.plate.upper())
    state = req.state.strip().upper()

    if not plate:
        raise HTTPException(status_code=400, detail="Missing license plate.")

    if not re.fullmatch(r"[A-Z]{2}", state):
        raise HTTPException(status_code=400, detail="State must be a 2-letter code like CA.")

    url_template = os.getenv("PLATE_LOOKUP_URL_TEMPLATE")
    api_key = os.getenv("PLATE_LOOKUP_API_KEY")

    if not url_template or not api_key:
        raise HTTPException(
            status_code=500,
            detail="Plate lookup is not configured on the server."
        )

    provider_url = url_template.format(
        plate=quote(plate),
        state=quote(state),
        api_key=quote(api_key),
    )

    try:
        request = Request(
            provider_url,
            headers={
                "Accept": "application/json",
                "User-Agent": "DiBuy/1.0",
            },
        )

        with urlopen(request, timeout=12) as response:
            raw = response.read().decode("utf-8")

        data = json.loads(raw)
        vehicle = _pick_vehicle_payload(data)

        return {
            "vin": _get_first(vehicle, "vin", "VIN"),
            "year": _get_first(vehicle, "year", "model_year", "ModelYear"),
            "make": _get_first(vehicle, "make", "Make"),
            "model": _get_first(vehicle, "model", "Model"),
            "trim": _get_first(vehicle, "trim", "submodel", "sub_model", "style", "body", "Trim"),
            "drivetrain": _get_first(vehicle, "drivetrain", "drive_type", "DriveType"),
            "transmission": _get_first(vehicle, "transmission", "transmission_style", "TransmissionStyle"),
            "vehicleType": _get_first(vehicle, "body_type", "body_style", "vehicle_type", "BodyClass"),
            "cylinders": _get_first(vehicle, "cylinders", "engine_cylinders", "EngineCylinders"),
        }

    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")[:300]
        raise HTTPException(
            status_code=502,
            detail=f"Plate provider returned {exc.code}: {detail}"
        )

    except URLError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach plate provider: {str(exc)[:200]}"
        )

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=502,
            detail="Plate provider returned invalid JSON."
        )

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Plate lookup failed: {type(exc).__name__}: {str(exc)[:200]}"
        )


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
        img = None

        try:
            img = Image.open(BytesIO(contents))
            img.load()
            del contents

            classification = classifier(img)

            nsfw_result = next(
                (r for r in classification if r["label"].lower() == "nsfw"),
                None
            )

            del classification

            if nsfw_result and nsfw_result["score"] >= NSFW_THRESHOLD:
                flagged.append(
                    FlaggedFile(
                        filename=file.filename,
                        reason="NSFW content detected",
                        score=nsfw_result["score"],
                    )
                )

        except HTTPException:
            raise

        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to open image {file.filename}: {exc}",
            )

        finally:
            if img is not None:
                img.close()
                del img

            gc.collect()

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

        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_output)

        if fence_match:
            raw_output = fence_match.group(1).strip()
        else:
            start = raw_output.find("{")
            end = raw_output.rfind("}")

            if start != -1 and end != -1:
                raw_output = raw_output[start: end + 1]

        parsed = json.loads(raw_output)

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
