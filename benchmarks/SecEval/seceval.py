import re
import sys
import json
import time
from tqdm import tqdm
from openai import OpenAI

class SevEvalEvaluator:

    def __init__(self, model_name, file_path, api_key="EMPTY", base_url="http://127.0.0.1:3000/v1"):

        # vLLM is OpenAI-compatible: just set base_url + any API key (vLLM ignores/accepts dummy keys)
        self.client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
        self.file_path = file_path
        self.model_name = model_name

    def read_json_file(self):
        with open(self.file_path, 'r') as file:
            return json.load(file)

    @staticmethod
    def extract_answer(response):
        if response and response.strip():
            m = re.search(r"ANSWER:?\s*([A-D])", response, re.IGNORECASE)
            return m.group(1).upper() if m else None
        return None
    
    # new
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
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=8,
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

    MODEL_NAME = sys.argv[-2]
    FILE_PATH = sys.argv[-1]

    evaluator = SevEvalEvaluator(
        file_path=FILE_PATH,
        model_name=MODEL_NAME
    )
    evaluator.run_evaluation()

