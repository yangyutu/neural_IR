import torch
from transformers import AutoModel

class BertEncoder(torch.nn.Module):
    def __init__(self, pretrained_model_name):
        super().__init__()
        self.bert_backbone = AutoModel.from_pretrained(pretrained_model_name)

    def forward(self, input):

        bert_output = self.bert_backbone(input_ids=input['input_ids'].squeeze(1), attention_mask=input['attention_mask'].squeeze(1))
        #bert_output is the last layer's hidden state
        last_hidden_state = bert_output.last_hidden_state
        cls_representation = last_hidden_state[:,0,:]
        return cls_representation

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BertEncoder")
        parser.add_argument("--pretrained_model_name", type=str, required=True)
        return parent_parser


if __name__ == '__main__':

    from transformers import AutoTokenizer

    name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = BertEncoder(name)
    text = ["Replace me by any text and words you'd like tokenization.", "Hello, word"]

    encoded_input = tokenizer(text, return_tensors='pt', padding=True)

    output = model(encoded_input)

