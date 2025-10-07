from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():

    model_id = "/data/nfs01/yichao/models/Qwen/Qwen3-8B"
    sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=16384)

    PROMPTS = [
        {"role": "user", "content": r"Evaluate the limit: \[ \lim_{x \to \infty} \sqrt{x} \left( \sqrt[3]{x+1} - \sqrt[3]{x-1} \right) \]\nPlease reason step by step, and put your final answer within \boxed{}."}
    ]

    tokenizer = AutoTokenizer.from_pretrained("/data/nfs01/yichao/models/Qwen/Qwen3-8B", trust_remote_code=True)
    inputs = [tokenizer.apply_chat_template(PROMPTS, tokenize=False, add_generation_prompt=True)]

    llm = LLM(model=model_id, tensor_parallel_size=1, disable_log_stats=False)
    # llm = LLM(model=model_id, tensor_parallel_size=1)
    outputs = llm.generate(inputs, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        print(f"Full output: {output}")

if __name__ == "__main__":
    main()