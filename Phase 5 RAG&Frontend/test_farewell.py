import re

FAREWELL_PATTERNS = [
    r"\bthat'?s all\b",
    r"^(okay|ok|alright|bye|goodbye|thanks|thank you|cheers|great|cool|got it|perfect|noted|sure|sounds good|see you|take care|done|nothing else|no more|all good|awesome|wonderful)[\s!.]*$",
]

test_phrases = [
    "okay that's all",
    "okay that",
    "that's all",
    "bye",
    "thanks",
    "ok",
    "what is 2+2",
    "tell me about ashwagandha",
]

def is_farewell(question):
    q = question.lower().strip()
    return any(re.search(p, q) for p in FAREWELL_PATTERNS)

for phrase in test_phrases:
    result = is_farewell(phrase)
    print(f"{'✅' if result else '❌'} '{phrase}' → farewell={result}")
