python get_scores.py --task countdown --input_file phi_outputs/phi_countdown_n25_mcl_256_outputs.jsonl --output_file scores/phi_countdown_n25_mcl_256_scores.jsonl
python get_scores.py --task countdown --input_file phi_outputs/phi_countdown_n25_mcl_1024_outputs.jsonl --output_file scores/phi_countdown_n25_mcl_1024_scores.jsonl
python get_scores.py --task countdown --input_file phi_outputs/phi_countdown_n50_mcl_256_outputs.jsonl --output_file scores/phi_countdown_n50_mcl_256_scores.jsonl
python get_scores.py --task countdown --input_file phi_outputs/phi_countdown_n50_mcl_1024_outputs.jsonl --output_file scores/phi_countdown_n50_mcl_1024_scores.jsonl
python get_scores.py --task countdown --input_file phi_outputs/phi_countdown_n100_mcl_256_outputs.jsonl --output_file scores/phi_countdown_n100_mcl_256_scores.jsonl
python get_scores.py --task countdown --input_file phi_outputs/phi_countdown_n100_mcl_1024_outputs.jsonl --output_file scores/phi_countdown_n100_mcl_1024_scores.jsonl
python get_scores.py --task countdown --input_file phi_outputs/phi_countdown_n25_mcl_256_quantized_outputs.jsonl --output_file scores/phi_countdown_n25_mcl_256_quantized_scores.jsonl
python get_scores.py --task countdown --input_file phi_outputs/phi_countdown_n50_mcl_256_quantized_outputs.jsonl --output_file scores/phi_countdown_n50_mcl_256_quantized_scores.jsonl
python get_scores.py --task countdown --input_file phi_outputs/phi_countdown_n100_mcl_256_quantized_outputs.jsonl --output_file scores/phi_countdown_n100_mcl_256_quantized_scores.jsonl
python get_scores.py --task aime --input_file phi_outputs/phi_aime_n89_mcl_256_outputs.jsonl --output_file scores/phi_aime_n89_mcl_256_scores.jsonl
python get_scores.py --task aime --input_file phi_outputs/phi_aime_n89_mcl_1024_outputs.jsonl --output_file scores/phi_aime_n89_mcl_1024_scores.jsonl
python get_scores.py --task countdown --input_file phi_outputs/phi_countdown_n100_mcl_256_pretrained_outputs.jsonl --output_file scores/phi_countdown_n100_mcl_256_pretrained_scores.jsonl
python get_scores.py --task aime --input_file phi_outputs/phi_aime_n89_mcl_256_pretrained_outputs.jsonl --output_file scores/phi_aime_n89_mcl_256_pretrained_scores.jsonl

