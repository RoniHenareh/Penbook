import os
import re
import json
import time
from tqdm import tqdm
#from openai import OpenAI 

# add reverse proxy instead
# start with: litellm --config litellm_config.yaml

# hide openai HTTP logging messages
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# better way to use API key
from dotenv import load_dotenv
load_dotenv()

# integrate LiteLLM
from openai import OpenAI

external_client = OpenAI(
  base_url = os.getenv('LITELLM_BASE_URL', 'http://localhost:4000'),
  api_key=os.getenv('LITELLM_API_KEY', 'key'))

# pick the model
#llm_model=os.getenv('LLM_MODEL', 'gpt-5')
#llm_model=os.getenv('LLM_MODEL', 'claude-4')
llm_model=os.getenv('LLM_MODEL', 'gemini-3')

# download models from ollama first
#llm_model=os.getenv('LLM_MODEL', 'deepseek-r1:8b')
#llm_model=os.getenv('LLM_MODEL', 'llama3.1:8b')
#llm_model=os.getenv('LLM_MODEL', 'mistral:7b')

class SecEvalEvaluator:

    def __init__(self, client, model, file_path):
        self.client = client
        self.model = model
        self.file_path = file_path

    def read_json_file(self):
        with open(self.file_path, 'r') as file:
            return json.load(file)

    @staticmethod
    def extract_answer(response):
        if response and response.strip():
            m = re.search(r"ANSWER:?\s*([A-D])", response, re.IGNORECASE)
            return m.group(1).upper() if m else None
        return None
    
    # new, choices
    @staticmethod
    def parse_choices(choices_list):
        """Convert choices list to dict format {A: text, B: text, ...}"""
        answers = {}
        for choice in choices_list:
            # Extract letter and text (format: "A: text" or "A. text")
            match = re.match(r'([A-D])[:\.\)]\s*(.*)', choice.strip())
            if match:
                letter = match.group(1).upper()
                text = match.group(2)
                answers[letter] = text
        return answers

    def ask_llm(self, question, answers, max_retries=5):
        options = ', '.join([f"{k}) {v}" for k, v in answers.items()])
        prompt = (
            f"Question: {question}\nOptions: {options}\n\n"
            "Choose the correct answer (A, B, C, or D) only.\n"
            "Return EXACTLY this format: ANSWER: X"
        )
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    #max_tokens=18,
                )
                if resp.choices:
                    result = self.extract_answer(resp.choices[0].message.content)
                    if result:
                        return result
                    else:
                        print("Incorrect answer format detected. Retrying...")
            except Exception as e:
                delay = 2 ** attempt
                print(f"Error: {e}. Retrying in {delay} s.")
                time.sleep(delay)
        return None

    def run_evaluation(self):
        data = self.read_json_file()
        # Handle both formats: direct list or nested under 'questions'
        questions = data if isinstance(data, list) else data.get('questions', [])

        correct = 0
        wrong = []

        with tqdm(total=len(questions), desc="Processing Questions") as bar:
            for item in questions:
                q = item['question']
                # New format: 'choices' as list, 'answer' for solution
                choices_list = item['choices']
                solution = item['answer']

                # Convert choices list to dict format
                ans = self.parse_choices(choices_list)

                guess = self.ask_llm(q, ans)
                if guess == solution:
                    correct += 1
                else:
                    wrong.append({"question": q, "correct_answer": solution, "llm_answer": guess})

                acc = correct / (bar.n + 1) * 100
                bar.set_postfix_str(f"Accuracy: {acc:.2f}%")
                bar.update(1)

        print(f"Final Accuracy: {correct / len(questions) * 100:.2f}%")
        if wrong:
            print("\nIncorrect Answers:")
            for w in wrong:
                print(f"Question: {w['question']}")
                print(f"Expected Answer: {w['correct_answer']}, LLM Answer: {w['llm_answer']}\n")

# Example usage:
if __name__ == "__main__":

    file_path='../SecEval/questions.json'

    evaluator = SecEvalEvaluator(client=external_client, model=llm_model, file_path=file_path)
    evaluator.run_evaluation()
