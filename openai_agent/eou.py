
import numpy as np
import string
from transformers import AutoTokenizer
import onnxruntime as ort
from huggingface_hub import hf_hub_download

def softmax(logits: np.ndarray) -> np.ndarray:
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

class EOU:
    def __init__(self):
        HG_MODEL = "livekit/turn-detector"
        ONNX_FILENAME = "model_quantized.onnx"

        local_path = hf_hub_download(repo_id=HG_MODEL, filename=ONNX_FILENAME, local_files_only=True)
        # print(local_path)

        self.session = ort.InferenceSession(local_path, providers=["CPUExecutionProvider"])

        self.tokenizer = AutoTokenizer.from_pretrained(
            HG_MODEL, local_files_only=True, truncation_side="left")

    def __call__(self,message):
        return self._calcEOU(message)

    def format_chat_messages(self,chat_ctx: dict):

        def normalize_text(text):

            PUNCS = string.punctuation.replace("'", "")

            def strip_puncs(text):
                return text.translate(str.maketrans("", "", PUNCS))

            return " ".join(strip_puncs(text).lower().split())

        new_chat_ctx = []
        for msg in chat_ctx:
            content = normalize_text(msg["content"])

            if not content:
                continue

            msg["content"] = content
            new_chat_ctx.append(msg)

        convo_text = self.tokenizer.apply_chat_template(
            new_chat_ctx,
            add_generation_prompt=False,
            add_special_tokens=False,
            tokenize=False,
        )

        # print("convo_text", convo_text)

        # remove the EOU token from current utterance
        ix = convo_text.rfind("<|im_end|>")
        text = convo_text[:ix]
        return text

    def calcMessagesEOU(self,messages, session, tokenizer):  
        MAX_HISTORY_TOKENS = 512

        text = self.format_chat_messages(messages)

        # print("text", text)

        inputs = tokenizer(
                text,
                return_tensors="np",
                truncation=True,
                max_length=MAX_HISTORY_TOKENS,
            )

        input_ids = np.array(inputs["input_ids"], dtype=np.int64)
        # Run inference
        outputs = session.run(["logits"], {"input_ids": input_ids})
        logits = outputs[0][0, -1, :]

        # Softmax over logits
        probs = softmax(logits)
        # The ID for the <|im_end|> special token
        eou_token_id = tokenizer.encode("<|im_end|>")[-1]
        return probs[eou_token_id]

    def _calcEOU(self, message):
        messages = [
            {"role": "user", "content": message}
        ]
        eou_prob = self.calcMessagesEOU(messages, self.session, self.tokenizer)

        return eou_prob
    
def create_eou():
    return EOU()
    

if __name__ == "__main__":
    from termcolor import colored

    eou = create_eou()

    def test(t):
        print(colored(t, "cyan"),colored("=>", "green"),colored(eou(t), "yellow"))

    test("Hello, how can I help you today?")
    test("I um am")
    test("I um am looking")
    test("I um am uh looking for a car.")

