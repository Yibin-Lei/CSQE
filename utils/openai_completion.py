import os
import time
import openai
from tqdm import tqdm
from time import sleep
import spacy


class OpenaiCompletion(object):
    MODELS = set(
        [
            "gpt-3.5-turbo-0301",
            "gpt-3.5-turbo",
            "gpt-4-0314",
            "gpt-4",
            "text-davinci-003",
            "text-davinci-002",
            "text-davinci-001",
        ]
    )

    def __init__(
            self,
            model_name="text-davinci-003",
            eos_token="<|endoftext|>",
            api_key=None
    ):
        assert model_name in self.MODELS

        self.model_name = model_name
        self.eos_token = eos_token

        if api_key is not None:
            openai.api_key = api_key
        self.api_key = openai.api_key

        self.spacy_nlp = None

    def completion_chat(self, messages, max_tokens=512, temperature=1, top_p=1, n=2,):
        def parse_api_result(result):
            to_return = []
            for idx, g in enumerate(result['choices']):
                to_return.append(g["message"]["content"])
            return to_return

        get_result = False
        res = []
        while not get_result:
            try:
                res = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    # stop=['\n\n\n', ],
                    max_tokens=max_tokens,
                    # logprobs=logprobs,
                )
                get_result=True
            except:
                sleep(1)
        return parse_api_result(res)

    def completion_non_chat(self, messages, max_tokens=512, temperature=1, top_p=1, n=2,):
        def parse_api_result(result):
            to_return = []
            for idx, g in enumerate(result['choices']):
                text = g['text']
                logprob = sum(g['logprobs']['token_logprobs'])
                to_return.append((text, logprob))
            to_return = [r[0] for r in sorted(to_return, key=lambda tup: tup[1], reverse=True)]
            return to_return


        get_result = False
        res = []
        while not get_result:
            try:
                print("input2llm" + messages[1]["content"])
                res = openai.Completion.create(
                    engine=self.model_name,
                    # messages=messages,
                    prompt=messages[1]["content"],
                    temperature=0.7,
                    top_p=top_p,
                    n=n,
                    stop=['\n\n\n', ],
                    max_tokens=max_tokens,
                    logprobs=1,
                )
                get_result=True
            except:
                sleep(1)
        return parse_api_result(res)

    def completion_multiple_with_scores(self, prompt, *args, **kwargs):
        if self.model_name in ["gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-4", "gpt-4-0314"]:
            return self.completion_multiple_with_scores_chat(prompt, *args, **kwargs)
        else:
            return self.completion_multiple_with_scores_instruct(prompt, *args, **kwargs)

    def completion_multiple_with_scores_instruct(
            self, prompt, max_tokens=512, temperature=0.7, top_p=1, n=2, logprobs=1,):
        def parse_api_result(result):
            to_return = []
            for idx, g in enumerate(result['choices']):
                text = g['text']
                logprob = sum(g['logprobs']['token_logprobs'])
                to_return.append((text, logprob))
            res = [r[0] for r in sorted(to_return, key=lambda tup: tup[1], reverse=True)]
            return res

        result = openai.Completion.create(
            engine=self.model_name,
            prompt=prompt,
            # api_key=self.api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stop=['\n\n\n', ],
            logprobs=logprobs,
        )
        return parse_api_result(result)

    def completion_multiple_with_scores_chat(
            self, prompt, max_tokens=512, temperature=0.7, top_p=1, n=2,):

        def parse_api_result(result):
            to_return = []
            for idx, g in enumerate(result['choices']):
                to_return.append(g["message"]["content"])
            return to_return

        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            top_p=top_p,
            n=n,
            # stop=['\n\n\n', ],
            max_tokens=max_tokens,
            # logprobs=logprobs,
        )
        return parse_api_result(res)

