import torch

def format_args_to_markdown_table(args):
    """
    Formats training arguments into a Markdown table.

    Args:
        args: An argparse.Namespace or similar object with attributes like
              nn_type, nn_batch_size, max_iters, lr, test, device, workers.

    Returns:
        A string containing the Markdown formatted table.
    """
    table_data = []

    table_data.append(("Model Type", args.nn_type))
    table_data.append(("Batch Size", args.nn_batch_size))
    table_data.append(("Epochs", args.max_iters))
    table_data.append(("Learning Rate", f"{args.lr:.0e}" if isinstance(args.lr, float) else args.lr))

    if args.test:
        table_data.append(("Data Used", "Test data"))
    else:
        table_data.append(("Data Used", "Train data"))

    device_info = "CPU only"
    if args.device != "cpu":
        try:
            if args.device == "cuda":
                if torch.cuda.is_available():
                    device_info = f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}"
                else:
                    device_info = "GPU (CUDA not available)"
            elif args.device.startswith("cuda:"):
                if torch.cuda.is_available():
                    device_id = int(args.device.split(':')[1])
                    if device_id < torch.cuda.device_count():
                        device_info = f"GPU: {torch.cuda.get_device_name(device_id)}"
                    else:
                        device_info = f"GPU: Invalid device ID {device_id}"
                else:
                    device_info = "GPU (CUDA not available)"
            else:
                device_info = f"Device: {args.device} (Specific GPU name lookup not applicable)"
        except Exception as e:
            device_info = f"GPU (Error fetching name: {e})"
            if not torch.cuda.is_available():
                device_info = "GPU (CUDA not available, or error)"


    table_data.append(("Device", device_info))
    table_data.append(("Workers", args.workers))

    markdown_table = "| Parameter         | Value                                  |\n"
    markdown_table += "| ----------------- | -------------------------------------- |\n"
    for key, value in table_data:
        if key == "" and value.startswith("**"):
            markdown_table += f"| {value:<17} |                                        |\n"
        else:
            markdown_table += f"| {key:<17} | {str(value):<38} |\n"

    return markdown_table