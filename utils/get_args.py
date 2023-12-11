

import argparse

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset_name", type=str, help="Evaluation dataset from the BEIR benchmark")
    parser.add_argument("--dataset_dir", type=str, default="./", help="Directory to save and load beir datasets")
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--model_path", type=str, help="Model path")
    parser.add_argument("--model_type", type=str, default="beir", help="beir or custom")

    parser.add_argument("--text_maxlength", type=int, default=512, help="Maximum text length")

    parser.add_argument("--corpus_chunk_size", default=50000, type=int, help="How many documents in one chunk,"
                                                                             "If memory is limited - make it smaller")

    parser.add_argument("--per_gpu_batch_size", default=128, type=int, help="Batch size per GPU/CPU for indexing.")
    # parser.add_argument("--output_dir", type=str, default="./my_experiment", help="Output directory")

    parser.add_argument("--norm_query", action="store_true", help="Normalize query representation")
    parser.add_argument("--norm_doc", action="store_true", help="Normalize document representation")
    parser.add_argument("--lower_case", action="store_true", help="lowercase query and document text")
    parser.add_argument("--normalize_text", action="store_true", help="Apply function to normalize some common characters")
    parser.add_argument(
        "--little_corpus", action="store_true", help="Use only a part of corpus (for debugging locally)")

    parser.add_argument("--log_dir", type=str, default="./logs/", help="Path to the log")
    parser.add_argument("--embedding_dir", type=str, default="./embeddings/", help="Path to the log")

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--main_port", type=int, default=-1, help="Main port (for multi-node SLURM jobs)")
    parser.add_argument("--hostname", type=str, default='gpunode-0-11',
                        help="hostname=$(scontrol show hostnames $ SLURM_JOB_NODELIST")

    parser.add_argument("--task", type=str, default="encode", help="encode/eval")
    parser.add_argument("--mask_ratio", type=float, default=0.1, help="The probability of masking, "
                                                             "required for query_alteration model selection method.")
    parser.add_argument("--topk", type=int, default=10, help="How many samples to extract"
                                                             "required for query_alteration model selection method."
                        )
    args, _ = parser.parse_known_args()
    return args