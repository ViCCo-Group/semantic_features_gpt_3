import os
import anthropic

client = anthropic.Client(os.environ['ANTHROPIC_API_KEY'])

def generate_single_prime_sentence(train_df, question):
    priming_text = f"{anthropic.HUMAN_PROMPT} Answer the last question. Use the same syntax."
    train_df = train_df[:3]
    for row in train_df.itertuples():
        priming_text += ' ' + row.question
        priming_text += ' '
        priming_text += row.answer

    text = f"{priming_text} {question}{anthropic.AI_PROMPT}" # See Claude API reference for format
    return text

def generate_chat_priming_messages(train_df, question):
    train_df = train_df[:3]
    text = ""
    for row in train_df.itertuples():
        text += f"{anthropic.HUMAN_PROMPT} {row.question}"
        text += f"{anthropic.AI_PROMPT} {row.answer}"

    text += f"{anthropic.HUMAN_PROMPT} {question} {anthropic.AI_PROMPT}"
    return text

def make_request(train_df, model, question):
    prompt = generate_single_prime_sentence(train_df, question)

    response = client.completion(
        prompt=prompt,
        stop_sequences = [anthropic.HUMAN_PROMPT],
        model="claude-v1",
        max_tokens_to_sample=80,
    )
    return response['completion']