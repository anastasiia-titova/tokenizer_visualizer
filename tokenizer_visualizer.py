from transformers import AutoTokenizer

# Tokenizer initialization
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Visualization function
def visualize_tokenization(text, tokenizer, model_name="GPT-2"):
    print(f"\nText: '{text}'")
    print(f"Tokenizer: {model_name}")

    tokens = tokenizer.encode(text, add_special_tokens=False)
    token_texts = [tokenizer.decode([t]) for t in tokens]

    print(f"\n{'ID':<10} | {'Token':<20} | {'Length':<5}")
    print(f"{'-' * 10} | {'-' * 20} | {'-' * 5}")

    for token_id, token_text in zip(tokens, token_texts):
        display_text = token_text.replace('Ġ', '[SP]').replace('\n', '[NL]').replace('\t', '[TB]')
        print(f"{token_id:<10} | {display_text:<20} | {len(token_text):<5}")

    print(f"\nStatistics:")
    print(f"   Tokens: {len(tokens)}")
    print(f"   Symbols: {len(text)}")
    if len(tokens) > 0:
        print(f"   Symbols per token: {len(text) / len(tokens):.2f}")

    return tokens, token_texts

# Test texts
test_texts = [
    "How to cook chicken for dinner?",
    "What time does the store close today?",
    "I need a recipe for chocolate cake",
    "Where can I buy fresh vegetables?",
    "How do I fix a leaky faucet?",
    "What should I wear for rainy weather?",
]

for text in test_texts:
    visualize_tokenization(text, tokenizer)

# Tokenization lenght comparison
print("\nComparison: number of tokens")

comparison_texts = [
    "I want to order pizza",
    "How to clean white sneakers at home",
    "What is the best way to brew coffee?",
    "I need to buy groceries for the week",
    "How do I reset my wifi password?",
    "Where is the nearest pharmacy open now?",
    "Can you help me plan a birthday party?",
    "What time does the bus arrive tomorrow?",
]

print(f"\n{'Phrase':<50} | {'Tokens':<10}")
print(f"{'-' * 50} | {'-' * 10}")

for text in comparison_texts:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    print(f"{text:<50} | {len(tokens):<10}")

# Demo BPE
print("\nDemo BPE")

bpe_examples = [
    "breakfast",
    "microwave",
    "refrigerator",
    "dishwasher",
    "airconditioner",
    "supercalifragilisticexpialidocious",
]

print("\nBPE breaks rare/long words into pieces:")
for word in bpe_examples:
    tokens = tokenizer.encode(word, add_special_tokens=False)
    token_texts = [tokenizer.decode([t]) for t in tokens]
    print(f"\n'{word}' → {len(tokens)} token(s): {token_texts}")

# Token calculator
print("\nToken calculator (valuation)")

def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))


everyday_queries = [
    "What should I cook for dinner tonight?",
    "How do I remove stains from white clothes?",
    "What time does the supermarket close on Sunday?",
    "Can you remind me to water the plants tomorrow?",
    "How do I fix a slow internet connection at home?",
    "What is a good workout routine for beginners?",
    "How do I plan a weekend trip to the mountains?",
    "What are some easy desserts I can make with kids?",
]

print("\nToken valuation:")
print(f"{'Query':<55} | {'Tokens':<8} | {'Price':<8}")
print(f"{'-' * 55} | {'-' * 8} | {'-' * 8}")

for text in everyday_queries:
    token_count = count_tokens(text, tokenizer)
    estimated_cost = (token_count / 1000) * 0.03
    display_text = text if len(text) <= 52 else text[:49] + "..."
    print(f"{display_text:<55} | {token_count:<8} | ${estimated_cost:.4f}")


