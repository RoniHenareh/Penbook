# base model
#python cybermetric.py "roni-finetuned-model" "/cephyr/users/ronih/Alvis/Desktop/sft/train/benchmarks/CyberMetric/CyberMetric-80-v1.json" 
#python cybermetric.py "roni-finetuned-model" "/cephyr/users/ronih/Alvis/Desktop/sft/train/benchmarks/CyberMetric/CyberMetric-500-v1.json" 
#python cybermetric.py "roni-finetuned-model" "/cephyr/users/ronih/Alvis/Desktop/sft/train/benchmarks/CyberMetric/CyberMetric-2000-v1.json" 
#python cybermetric.py "roni-finetuned-model" "/cephyr/users/ronih/Alvis/Desktop/sft/train/benchmarks/CyberMetric/CyberMetric-10000-v1.json"

# fine tuned model
python cybermetric.py "checkpoint-36"  "/cephyr/users/ronih/Alvis/Desktop/sft/train/benchmarks/CyberMetric/CyberMetric-80-v1.json"
python cybermetric.py "checkpoint-36" "/cephyr/users/ronih/Alvis/Desktop/sft/train/benchmarks/CyberMetric/CyberMetric-500-v1.json" 
python cybermetric.py "checkpoint-36" "/cephyr/users/ronih/Alvis/Desktop/sft/train/benchmarks/CyberMetric/CyberMetric-2000-v1.json" 
#python cybermetric.py "checkpoint-" "/cephyr/users/ronih/Alvis/Desktop/sft/train/benchmarks/CyberMetric/CyberMetric-10000-v1.json
