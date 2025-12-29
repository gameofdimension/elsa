# ref https://github.com/sgl-project/mini-sglang/blob/7f896d198fca46b392bc8faed7a65a43b3aacc37/benchmark/offline/bench.py#L1
import sys
import time

from minisgl.core import SamplingParams
from minisgl.llm import LLM
from prompt import text as aprompt


def main():
    count = int(sys.argv[1])

    model_path = "Qwen/Qwen3-1.7B"
    # llm = LLM(model_path=model_path)
    llm = LLM(model_path=model_path, max_seq_len_override=4096, max_extend_tokens=16384, cuda_graph_max_bs=256)
    max_tokens = 7  # real generated tokens is max_tokens - 2
    params = SamplingParams(max_tokens=max_tokens, temperature=0.9)

    n = 20
    prompt_lst = [aprompt for _ in range(n)]
    params_lst = [params for _ in range(n)]
    outputs = llm.generate(prompt_lst, sampling_params=params_lst)

    begin = time.time()
    for i in range(count):
        outputs = llm.generate(prompt_lst, sampling_params=params_lst)
        print(i, len(outputs), len(outputs[0]["token_ids"]))
    end = time.time()

    print(f"generate count {count}, mean time {(end - begin) / count:.3f}")


if __name__ == '__main__':
    main()

