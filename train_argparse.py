# coding=utf-8
import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Chinese Spell Correct')

    parser.add_argument("--init_bert", type=str, choices=["cbert", "bert", "macbert", "ernie3"], default="cbert",
                        help='the weights of model init type')

    # dataset path options
    parser.add_argument("--train_data_path", type=str,
                        help='path of the train dataset')
    parser.add_argument("--test_data_path", type=str,
                        help='path of the test dataset')
    
    # bert config options
    parser.add_argument("--config_path", default=None, type=str,
                        help="Path of the config file.")

    # tokenizer options
    parser.add_argument("--vocab_path", default=None, type=str,
                        help="Path of the vocabulary file.")
    
    # model options
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default=None, type=str,
                        help="Path of the output model.")
    parser.add_argument("--from_tf", action="store_true", default=False,
                        help="whether to load params from tensorflow.")

    # confusions options
    parser.add_argument("--confusions", default=None, type=str,
                        help="Path of confusions data")

    # data options
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="Batch size of train dataset.")
    parser.add_argument("--test_batch_size", type=int, default=32,
                        help="Batch size of test dataset.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout.")
    
    # optimization options
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warmup ratio value.")
    parser.add_argument("--optimizer", choices=["adamw", "adafactor"],
                        default="adamw",
                        help="Optimizer type.")
    parser.add_argument("--scheduler", choices=["linear", "cosine", "cosine_with_restarts", "polynomial",
                                                "constant", "constant_with_warmup"],
                        default="linear", help="Scheduler type.")

    # trainer options
    parser.add_argument('--accelerator', type=str, choices=["cpu", "gpu", "tpu", "ipu", "auto"], default='auto',
                        help='different accelerator types')
    parser.add_argument('--num_devices', type=int, default=1,
                        help='the number of devices of one node for training')
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='the number of node for training')
    parser.add_argument('--strategy', type=str, 
                        choices=["dp", "ddp", "ddp_spawn", "deepspeed", 'deepspeed_stage_1', 'deepspeed_stage_2', 'deepspeed_stage_3', "ddp_sharded"], 
                        default='ddp',
                        help='Strategy for how to run across multiple devices')
    parser.add_argument("--fp16", action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit.")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of epochs.")
    parser.add_argument("--gradient_clip_val", default=1.0, type=float,
                        help="gradient clip norm val.")
    parser.add_argument('--grad_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--val_check_interval', type=int, default=1000,
                        help="How often within one training epoch to check the validation set")

    # report step
    parser.add_argument("--report_steps", type=int, default=200,
                        help="Specific steps to print prompt.")

    # metric options
    parser.add_argument('--ignore_index', type=int, default=-100,
                        help='metric ignore index')
    
    # evn options
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    args = parser.parse_args()
    return args