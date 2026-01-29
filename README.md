# Fine-Tuning Small Open-Source LLMs for Cybersecurity

> **Master's Thesis** · Degree Programme in Engineering Mathematics  
> KTH Royal Institute of Technology, Stockholm  
> January 2026

**Author:** Roni Henareh

---

## Abstract

Running foundation models through an API can become costly at scale. Smaller models, on the other hand, are more cost-efficient and give organizations full control over their data while reducing the risk of breaches and ensuring compliance with regulations. Foundation models like GPT-5, Gemini-3 and Claude-4.5 are massive, estimated to be near 1 trillion parameters. This thesis compares the performance of fine-tuned small open-weight LLMs with foundation models on available Cybersecurity benchmarks.

Key insights include when fine-tuning should be used and when to consider other techniques like RAG or prompt engineering. Furthermore, this thesis also explores different techniques for training models, from full-parameter fine-tuning such as continued pre-training to parameter-efficient fine-tuning such as LoRA and QLoRA. Finally, the results of supervised fine-tuning on relevant Cybersecurity benchmarks are presented.

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{henareh2026finetuning,
  author  = {Henareh, Roni},
  title   = {Fine-Tuning Small Open-Source LLMs for Cybersecurity},
  school  = {KTH Royal Institute of Technology},
  year    = {2026},
  address = {Stockholm, Sweden},
  month   = {January}
}
```

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## Acknowledgments

First, I extend my appreciation to Cybercampus Sweden for providing me with the opportunity and resources to conduct this master's thesis work. I would especially like to thank David Olgart, the director of Cybercampus Sweden.

This research would not have been possible without access to the computational infrastructure provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS).

I would also like to thank all the students in the Royal Hacking Lab that I had the privilege to work besides during my time there. Moreover, I extend my sincere thanks to my academic supervisor and examiner, Emre Süren and Pontus Johnson — I am grateful for your encouragement and belief in my work.

Finally, I would like to express my gratitude to my family for their support and encouragement throughout my studies. Without them, this journey would not have been possible. Thank you all for your invaluable contribution.
