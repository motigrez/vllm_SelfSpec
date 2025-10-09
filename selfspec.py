from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time

def main():

    model_id = "Qwen/Qwen3-32B"
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=16384)

    PROMPTS = [
        [{"role": "user", "content": r"Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.\nPlease reason step by step, and put your final answer within \boxed{}."}],
        [{"role": "user", "content": r"Let $ABC$ be a triangle inscribed in circle $\omega$. Let the tangents to $\omega$ at $B$ and $C$ intersect at point $D$, and let $\overline{AD}$ intersect $\omega$ at $P$. If $AB=5$, $BC=9$, and $AC=10$, $AP$ can be written as the form $\frac{m}{n}$, where $m$ and $n$ are relatively prime integers. Find $m + n$.\nPlease reason step by step, and put your final answer within \boxed{}."}],
        [{"role": "user", "content": r"Each vertex of a regular octagon is independently colored either red or blue with equal probability. The probability that the octagon can then be rotated so that all of the blue vertices end up at positions where there were originally red vertices is $\tfrac{m}{n}$, where $m$ and $n$ are relatively prime positive integers. What is $m+n$?\nPlease reason step by step, and put your final answer within \boxed{}."}],
        [{"role": "user", "content": r"Define $f(x)=|| x|-\tfrac{1}{2}|$ and $g(x)=|| x|-\tfrac{1}{4}|$. Find the number of intersections of the graphs of \[y=4 g(f(\sin (2 \pi x))) \quad\text{ and }\quad x=4 g(f(\cos (3 \pi y))).\]\nPlease reason step by step, and put your final answer within \boxed{}."}],
        [{"role": "user", "content": r"Let $p$ be the least prime number for which there exists a positive integer $n$ such that $n^{4}+1$ is divisible by $p^{2}$. Find the least positive integer $m$ such that $m^{4}+1$ is divisible by $p^{2}$.\nPlease reason step by step, and put your final answer within \boxed{}."}],
        [{"role": "user", "content": r"Let $ABCD$ be a tetrahedron such that $AB=CD= \sqrt{41}$, $AC=BD= \sqrt{80}$, and $BC=AD= \sqrt{89}$. There exists a point $I$ inside the tetrahedron such that the distances from $I$ to each of the faces of the tetrahedron are all equal. This distance can be written in the form $\frac{m \sqrt n}{p}$, where $m$, $n$, and $p$ are positive integers, $m$ and $p$ are relatively prime, and $n$ is not divisible by the square of any prime. Find $m+n+p$.\nPlease reason step by step, and put your final answer within \boxed{}."}],
        [{"role": "user", "content": r"Let $\mathcal{B}$ be the set of rectangular boxes with surface area $54$ and volume $23$. Let $r$ be the radius of the smallest sphere that can contain each of the rectangular boxes that are elements of $\mathcal{B}$. The value of $r^2$ can be written as $\frac{p}{q}$, where $p$ and $q$ are relatively prime positive integers. Find $p+q$.\nPlease reason step by step, and put your final answer within \boxed{}."}],
        [{"role": "user", "content": r"There exist real numbers $x$ and $y$, both greater than 1, such that $\log_x\left(y^x\right)=\log_y\left(x^{4y}\right)=10$. Find $xy$.\nPlease reason step by step, and put your final answer within \boxed{}."}]
    ]

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    inputs = [tokenizer.apply_chat_template(PROMPTS, tokenize=False, add_generation_prompt=True)]
    temp = inputs[0]
    inputs = [x for x in temp for _ in range(8)]
    prompt_group_ids = [i // 8 + 1 for i in range(len(inputs))]

    llm = LLM(
        model = model_id, 
        tensor_parallel_size=1, 
        disable_log_stats=False, 
        scheduling_policy="priority", 
        gpu_memory_utilization=0.5,
        max_num_seqs=256,
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": 4,
            "prompt_lookup_max": 6,
        },
        compilation_config={  # 新增部分
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 576, 640, 768, 1024]
        }
    )
    
    
    # llm = LLM(model=model_id, tensor_parallel_size=1)
    print("Start generating...")

    start_time = time.time()
    outputs = llm.generate(inputs, sampling_params, prompt_group_ids=prompt_group_ids)
    end_time = time.time()

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    duration = end_time - start_time
    throughput = total_tokens / duration
    print(f"Generated {total_tokens} tokens in {duration:.2f} seconds. Throughput: {throughput:.2f} tokens/second.")

if __name__ == "__main__":
    main()