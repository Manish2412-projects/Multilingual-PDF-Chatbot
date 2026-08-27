from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# List all available models
models = client.models.list()

print("✅ Models available on your account:")
print("─────────────────────────────────────")
for model in models.data:
    print(f"➜ {model.id}")