import torch
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from shark_runner import shark_inference

torch.manual_seed(0)
tokenizer = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")


def _prepare_sentence_tokens(sentence: str):
    return torch.tensor([tokenizer.encode(sentence)])


class MiniLMSequenceClassification(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/MiniLM-L12-H384-uncased",  # The pretrained model.
            num_labels=2,  # The number of output labels--2 for binary classification.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
            torchscript=True,
        )

    def forward(self, tokens):
        return self.model.forward(tokens)[0]


test_input = _prepare_sentence_tokens("this project is very interesting")


results = shark_inference(
    MiniLMSequenceClassification(), test_input, device="cpu", jit_trace=True
)
