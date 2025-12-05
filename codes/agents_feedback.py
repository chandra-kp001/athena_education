import requests
import json
import re
def extract_json(text):
    """
    Safely extracts the first JSON object from a model output.
    Handles markdown, code fences, extra text, etc.
    """
    if not text or text.strip() == "":
        return {"error": "Empty model output", "raw": text}

    # Remove ```json and ``` wrappers if present
    cleaned = text.replace("```json", "").replace("```", "").strip()

    # Extract JSON between { ... }
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            return {"error": "Malformed JSON", "raw": cleaned}

    return {"error": "No JSON found", "raw": cleaned}
def print_json(data):
    """Pretty-prints a Python dictionary as JSON."""
    print(json.dumps(data, indent=4, ensure_ascii=False))
def run_shark_analysis(transcript_text,api_key):
    API_KEY = api_key
    url = "https://api.deepseek.com/v1/chat/completions"
    sharks = {
        "Visionary Shark": """
        You focus on:
        - market potential
        - innovation
        - long-term scalability
        - disruptive ideas
        """,

        "Finance Shark": """
        You focus on:
        - revenue model clarity
        - margins & unit economics
        - financial feasibility
        - monetization strength
        """,

        "Skeptic Shark": """
        You focus on:
        - assumptions that seem unrealistic
        - weaknesses or missing details
        - risks the founder ignored
        """,

        "Customer Advocate Shark": """
        You focus on:
        - problem clarity
        - user pain points
        - how well the solution helps real customers
        """
    }

    final_output = {}

    for shark_name, shark_focus in sharks.items():

        prompt = f"""
You are *{shark_name}*.

Your perspective:
{shark_focus}

Your job:
1. Read the transcript.
2. Evaluate ONLY from your shark perspective.
3. Output STRICT JSON format:
{{
  "feedback": "5–8 line narrative feedback",
  "strengths": ["one", "two", "three"],
  "weaknesses": ["one", "two", "three"],
  "verdict": "Invest / Not Invest / Need More Info"
}}

Transcript:
\"\"\" 
{transcript_text}
\"\"\"

Return ONLY valid JSON.
"""

        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5
        }

        response = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {API_KEY}"}
        )

        model_output = response.json()["choices"][0]["message"]["content"]
        # Convert JSON string to Python dict
        shark_json = extract_json(model_output)
        final_output[shark_name] = shark_json
        #print_json(shark_json)
    return final_output
if __name__ == "__main__":
    transcript_text = input("Enter transcript text: ")
    api_key = input("Enter API key: ")
    result = run_shark_analysis(transcript_text,api_key)
    print(result)