# base model
python seceval.py "roni-finetuned-model" "/cephyr/users/ronih/Alvis/Desktop/sft/train/benchmarks/SecEval/questions.json" 

# fine tuned model
python seceval.py "checkpoint-36"  "/cephyr/users/ronih/Alvis/Desktop/sft/train/benchmarks/SecEval/questions.json"
