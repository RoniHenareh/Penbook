import sys
import wandb
import torch
from huggingface_hub import login
from trl import SFTTrainer, SFTConfig
from transformers import BitsAndBytesConfig
from dataclasses import dataclass, field, asdict
from peft import LoraConfig as PEFTLoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

# format data
from data.format1 import format_data

login(token="") # HF
wandb.login(key="") # wandb

@dataclass 
class ModelConfig:

    # Transformers configuration
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"

    # LoRA
    r: int = 64 # rank
    lora_alpha: int = 128 # r * 2
    lora_dropout: float = 0.1 # 0.1 - 0.5

    task_type: str = "CAUSAL_LM"
    target_modules: str = "all-linear" # testa, to much memory
    #target_modules: tuple[str] = ("q_proj", "k_proj", "v_proj", "o_proj")

@dataclass
class DataConfig:

    dataset_name: str = "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset" #sys.argv[-3] # Given at runtime
    
    run_name: str = "HackMentorInstruct"
    wandb_project: str = "Stacked-Instruct"

    val_size: float = 0.2 # 20% validation

@dataclass 
class SFT:

    # Hyperparameters For Training

    # sequence length, chunk for tokenizer
    max_length: int = 2048

    output_dir: str = sys.argv[-1] # Given at runtime

    # effective batch size = 3840
    # gradient_accumulation_steps = 240, per_device_train_batch_size = 16
    per_device_eval_batch_size: int = 16 # trick, 1
    per_device_train_batch_size: int = 16 # trick, 1
    gradient_accumulation_steps: int = 240 # trick, 3840

    # Memory-efficient optimizer, esp. with quantization
    optim: str = "paged_adamw_8bit" 

    max_grad_norm: float = 0.3 # norm i algebra ||g|| (L2 norm) här, 
    # om ||g|| större än 0.3: skalfaktor = max_grad_norm / ||g||

    warmup_ratio: float = 0.03  # another way to calculate warmup steps
    # Warmup steps = total_steps * warmup_ratio, where:
    # total_steps = steps_per_epoch * num_train_epochs

    #fp16: bool = True # for older GPUs (V100, T4, etc.)
    bf16: bool = True # for newer GPUs A100/H100

    # higher batch sizes with lower learning rates is good according to paper
    learning_rate: float = 2e-5 # learning rate from secret recepie paper
    lr_scheduler_type: str = "cosine" # linear / cosine

    # AdamW integrates L2 regularisation
    weight_decay: float = 0.0 # L2 regularization?
    num_train_epochs: int = 3 # 2-3, otherwise overfitting

    # Save Model And Report

    # save every N = datapoint/batch size
    eval_strategy: str = "steps" 
    save_strategy: str = "steps" 

    # save often
    eval_steps: int = 8 # 200 before
    save_steps: int = 8 # 200 before

    # save only best checkpoint
    save_total_limit: int = 2 # keep last + best

    # lower loss = better
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False 

    # loss frequently in wandb
    logging_steps: int = 1 # 20 before

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sft: SFT = field(default_factory=SFT)

def main(cfg: Config):

    wandb.init(
        project=cfg.data.wandb_project,
        name=cfg.data.run_name,
        job_type="train",
        config=asdict(cfg),
    )

    # Load Model and Tokenizer
        
    # detect gpu if available
    device = 'cpu'
    if torch.cuda.is_available(): 
        device = 'cuda'    
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print("CUDA available:", torch.cuda.is_available())
        print("GPU name:", torch.cuda.get_device_name(0))
        #print("GPU memory summary:\n", torch.cuda.memory_summary())

    dtype_option = torch.float16
    if torch.cuda.is_bf16_supported():
        dtype_option = torch.bfloat16

    # inspo from MedGemma
    model_kwargs = dict(
        attn_implementation="eager",
        dtype=dtype_option, # `torch_dtype` is deprecated! Use `dtype` instead!
        device_map="auto", # alt to #.to(device_option)
    ) 

    # QLoRA
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, # memory saving
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=model_kwargs["dtype"], # `torch_dtype` is deprecated! Use `dtype` instead!
        bnb_4bit_quant_storage=model_kwargs["dtype"], # `torch_dtype` is deprecated! Use `dtype` instead!
    )

    # Model
    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_id, **model_kwargs)#.to(device)

    print('Optimize LoRA: ', model)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA Config
    lora_config = PEFTLoraConfig(
        r=cfg.model.r,
        task_type=cfg.model.task_type,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        target_modules=cfg.model.target_modules
    )
    print('printing trainable parameters: ')
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
  
    # Load and Format Dataset
    #format_func = func_map[cfg.data.dataset_name]
    dataset = format_data(cfg.data.dataset_name)

    # Split dataset into training/testing
    dataset = dataset.train_test_split(test_size=cfg.data.val_size, shuffle=True)
    training_dataset = dataset["train"]
    testing_dataset = dataset["test"]

    # Use the test split as the validation set
    dataset["validation"] = dataset.pop("test")

    # Display dataset details
    print(dataset)

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # new, make space
    torch.cuda.empty_cache()

    # Initialize trainer
    training_args = SFTConfig(

        # new, make space
        gradient_checkpointing=True,

        max_length=cfg.sft.max_length,
        output_dir=cfg.sft.output_dir,

        per_device_train_batch_size=cfg.sft.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.sft.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.sft.gradient_accumulation_steps,

        optim=cfg.sft.optim,

        max_grad_norm=cfg.sft.max_grad_norm,

        warmup_ratio=cfg.sft.warmup_ratio,

        #fp16=cfg.sft.fp16,
        bf16=cfg.sft.bf16,

        learning_rate=cfg.sft.learning_rate,
        lr_scheduler_type=cfg.sft.lr_scheduler_type,

        weight_decay = cfg.sft.weight_decay,

        num_train_epochs=cfg.sft.num_train_epochs,

        eval_strategy = cfg.sft.eval_strategy,
        save_strategy = cfg.sft.save_strategy,

        eval_steps=cfg.sft.eval_steps,
        save_steps=cfg.sft.save_steps,

        save_total_limit = cfg.sft.save_total_limit,

        # lower loss = better
        metric_for_best_model = cfg.sft.metric_for_best_model,
        greater_is_better = cfg.sft.greater_is_better,

        # wandb
        logging_steps = cfg.sft.logging_steps,
        run_name = cfg.data.run_name, 
        report_to=["wandb"],
    )

    trainer = SFTTrainer( 
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=testing_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        data_collator=data_collator,
        )
    # Train and Save model
    trainer.train()

if __name__ == "__main__":
    cfg = Config() 
    main(cfg)
