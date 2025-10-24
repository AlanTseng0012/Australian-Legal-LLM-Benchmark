from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

#MODEL = "gpt-5-mini"
#THINKING_BUDGET = 2048 #gpt-5

MODEL = "gpt-4o-mini"
OUTPUT_CAP_TOKENS = 512      # for visible answer

enc = tiktoken.get_encoding("cl100k_base")

def truncate_tokens(text: str, max_tokens: int) -> str:
    ids = enc.encode(text)
    if len(ids) <= max_tokens:
        return text
    return enc.decode(ids[:max_tokens])

def main():
    load_dotenv()
    print("=========== Chat started. Type 'exit' to quit. ===========")

    client = OpenAI()

    while True:
        user = input("\nYou: ").strip()
        if not user:
            continue
        if user.lower() in {"exit"}:
            break

        req = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": user + "\n\nReturn the final answer succinctly; <512 tokens."
                }
            ],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": OUTPUT_CAP_TOKENS,
            #"max_completion_tokens": THINKING_BUDGET, #gpt-5
            #"verbosity": "low" #gpt-5
        }

        try:
            r = client.chat.completions.create(**req)
            choice = r.choices[0]
            reply = (choice.message.content or "").strip() if choice.message else ""
            reply = truncate_tokens(reply, OUTPUT_CAP_TOKENS)
            print(f"LLM: {reply or '[empty reply]'}")
        except Exception as e:
            print(f"[error] {e}")

if __name__ == "__main__":
    main()
