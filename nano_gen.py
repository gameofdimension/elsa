from nanovllm import LLM, SamplingParams

from prompt import text as aprompt


def main(ckpt):
    llm = LLM(ckpt, enforce_eager=True, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.6, max_tokens=5, ignore_eos=True)
    n = 20
    prompt_lst = [aprompt for _ in range(n)]
    params_lst = [sampling_params for _ in range(n)]
    outputs = llm.generate(prompt_lst, params_lst)
    # outputs[0]["text"]
    print(outputs[0]["text"])


if __name__ == '__main__':
    ckpt = "Qwen/Qwen3-1.7B"
    main(ckpt)
