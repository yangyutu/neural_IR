import torch


def encode_plus(token, tokenizer, max_len=128):
    encoded = tokenizer.encode_plus(
        token,
        return_tensors="pt",
        max_length=max_len,
        truncation="longest_first",
        padding="max_length",
    )
    return encoded


def batch_inference(model, encodings, device, batch_size=128):

    output = []
    for start in range(0, len(encodings), batch_size):
        end = start + batch_size
        batch_input = {
            key: [i[key] for i in encodings[start:end]] for key in encodings[0]
        }
        batch_input_cuda = {k: torch.cat(v).to(device) for k, v in batch_input.items()}
        batch_out = model(batch_input_cuda)
        output.append(batch_out)

    output = torch.cat(output)
    return output