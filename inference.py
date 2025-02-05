from unsloth import FastLanguageModel

#Model config
max_seq_length = 2048
dtype = None
load_in_4bit = True

model,tokenizer = FastLanguageModel.from_pretrained(
    model_name = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = userdata.get('HF_TOKEN')
)
#sample question and prompt like the data we have
prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.
### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.
### Question:
{}

### Response:
<think>
{}
</think>"""
question = "A 40-year-old female presents with fever, fatigue, and diffuse painful swelling in the midline of the neck. Fine needle aspiration cytology (FNAC) reveals epithelioid cells and giant cells. Based on these clinical and cytological findings, what is the most likely diagnosis?"

#Inference Config
FastLanguageModel.for_inference(model)
input = tokenizer([prompt_style.format(question,"")],return_tensors='pt').to('cuda')
outputs = model.generate(
    input_ids = input.input_ids,
    attention_mask = input.attention_mask,
    max_new_tokens = 1200,
    eos_token_id = tokenizer.eos_token_id,
    use_cache = True
)
response = tokenizer.batch_decode(outputs,skip_special_tokens=True)
print(response[0].split("### Response:")[1])
