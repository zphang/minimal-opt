import argparse
import minimal_opt
import torch
from tqdm import auto as tqdm_lib
torch.set_grad_enabled(False)


def run_step(model, batch_size, input_len, output_len):
    input_ids = torch.zeros(batch_size, input_len).long().cuda()
    max_seq_len = input_len + output_len

    initial_input_length = input_ids.shape[1]
    current_input_ids = input_ids
    layer_past = None
    layer_past_length = 0
    all_token_ids = input_ids.tolist()
    batch_size = len(all_token_ids)

    trange = range(initial_input_length, max_seq_len)
    with torch.inference_mode():
        for _ in trange:
            input_length = current_input_ids.shape[1]
            model_out, layer_past = model(
                current_input_ids,
                layer_past=layer_past,
            )
            greedy_predicted_token_ids = model_out[:, -1].argmax(-1)
            current_input_ids = greedy_predicted_token_ids[:, None]
            for i in range(batch_size):
                all_token_ids[i].append(greedy_predicted_token_ids[i])
            layer_past_length += input_length


def create_model(model_name):
    config = {
        "125m": minimal_opt.OPT_125M_CONFIG,
        "1.3b": minimal_opt.OPT_1_3B_CONFIG,
        "2.7b": minimal_opt.OPT_2_7B_CONFIG,
        "6.7b": minimal_opt.OPT_6_7B_CONFIG,
        "13b": minimal_opt.OPT_13B_CONFIG,
        "30b": minimal_opt.OPT_30B_CONFIG,
        "66b": minimal_opt.OPT_66B_CONFIG,
        "175b": minimal_opt.OPT_175B_CONFIG,
    }[model_name]
    model = minimal_opt.PPOPTModel(config, use_cache=True)
    return model


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--input_len', type=int, default=1024)
    parser.add_argument('--output_len', type=int, default=128)
    parser.add_argument('--num_steps', type=int, default=10)
    args = parser.parse_args()

    model = create_model(args.model_name)
    run_step(
        model=model,
        batch_size=args.batch_size,
        input_len=args.input_len,
        output_len=args.output_len,
    )
    for _ in tqdm_lib.trange(args.num_steps):
        run_step(
            model=model,
            batch_size=args.batch_size,
            input_len=args.input_len,
            output_len=args.output_len,
        )
    print(f"{args.batch_size} Done.")


if __name__ == "__main__":
    main()
