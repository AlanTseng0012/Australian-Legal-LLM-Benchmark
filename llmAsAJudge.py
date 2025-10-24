import os
from dotenv import load_dotenv
from openai import OpenAI

# ---------- load .env ----------
load_dotenv()

# ---------- setup ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- prompt template ----------
RUBRIC_PROMPT = """
You are a strict evaluator in Legal content. Compare the following two answers and give a score between 0 and 1.

Rubric:
- 1.0 = Perfect match 
   (e.g., answer reproduces all key legal principles completely and accurately).
- 0.7~1.0 = Mostly correct, small gaps 
   (e.g., identifies correct duty but omits a detail like timing or scope).
- 0.4~0.7 = Partially correct 
   (e.g., mentions the correct Act but misstates some provisions or misses key principles).
- 0.1~0.4 = Minimal correctness 
   (e.g., vague reference such as "duty of good faith" without explanation).
- 0~0.1 = Incorrect 
   (e.g., talks about unrelated law such as property rights).

Instructions:
- Only return a numeric score between 0 and 1.
- Do not explain, just output the score.

Answer A (ground truth):
{ground_truth}

Answer B (model output):
{model_answer}
"""

def score_short_answer(ground_truth: str, model_answer: str) -> float:
    prompt = RUBRIC_PROMPT.format(
        ground_truth=ground_truth,
        model_answer=model_answer
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw_output = response.choices[0].message.content.strip()
    try:
        return float(raw_output)
    except ValueError:
        raise ValueError(f"Unexpected LLM output: {raw_output}")

if __name__ == "__main__":
    print("Type 'exit' anytime to quit.\n")
    while True:
        gt = input("Enter ground truth answer (or 'exit'): ").strip()
        if gt.lower() == "exit":
            break
        ma = input("Enter model answer (or 'exit'): ").strip()
        if ma.lower() == "exit":
            break
        try:
            score = score_short_answer(gt, ma)
            print(f"Score: {score}\n")
        except Exception as e:
            print(f"Error: {e}\n")
