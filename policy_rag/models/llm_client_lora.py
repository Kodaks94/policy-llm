from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import torch

class LoRALLMClient:
    """
    Hugging Face LLM client using a LoRA-adapted model for lightweight fine-tuning.
    Compatible with the main answerer module.
    """

    def __init__(self,
                 base_model="microsoft/phi-1_5",
                 lora_model_path="./lora_policy_adapter",
                 device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load LoRA configuration and base model
        print(f"🔹 Loading base model: {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        try:
            # If a LoRA adapter directory exists, load it
            config = PeftConfig.from_pretrained(lora_model_path)
            base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32)
            self.model = PeftModel.from_pretrained(base, lora_model_path)
            print(" Loaded LoRA adapter successfully.")
        except Exception as e:
            print(" Could not find LoRA adapter, using base model instead:", e)
            self.model = AutoModelForCausalLM.from_pretrained(base_model)

        self.model.to(self.device)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0 if self.device == "cuda" else -1)

    def generate(self, prompt: str, max_new_tokens=128, temperature=0.7, top_p=0.95):
        """Generate text using the LoRA-augmented model."""
        response = self.pipe(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, do_sample=True)
        return response[0]["generated_text"]



#training phase
'''
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/phi-1_5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # adapt attention layers
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)

# Fine-tune on your policy dataset
# (list of instruction-response pairs or plain documents)
# Then save:
model.save_pretrained("./lora_policy_adapter")
'''
