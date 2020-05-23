# Language Model Fine-tuning
Prepare data for language model fine-tuning.

#### Fine-tuning on Wiki paragraphs:

1. Prepare text paragraphs:

   ```bash
   python prepare_wiki_finetune.py -data_dir ../data\
   -wiki ../wiki/wiki_para_50.json -output_dir ../finetune_data
   ```

2. Find nouns and verbs from text paragraphs and store their mention indices:

   ```bash
   python extract_wiki_noun_verb.py -train_file ../finetune_data/train.txt\
   -eval_file ../finetune_data/eval.txt
   ```

   The generated files are `train.txt` (train text), `eval.txt` (eval text), `train.pos` (train noun/verb indices), `eval.pos` (eval noun/verb indices)

3. Fine-tune a BERT model using `../run_lm.py`

   ```bash
   python run_lm.py --model_type bert --model_name_or_path bert-base-uncased\
   --do_train --train_data_file finetune_data/train.txt --train_pos_file finetune_data/train.pos\
   --do_eval --eval_data_file finetune_data/eval.txt --eval_pos_file finetune_data/eval.pos\
   --mlm --mlm_probability 0.3 --line_by_line --num_train_epochs 5 --warmup_steps 750\
   --logging_steps 750 --save_steps 750 --eval_all_checkpoints --per_gpu_train_batch_size 8\
   --gradient_accumulation_steps 2 --per_gpu_eval_batch_size 16 --learning_rate 5e-5  --output_dir OUTPUT_DIR
   ```
   
   where `OUTPUT_DIR` should be the directory to store the saved models.

#### Fine-tuning on ConceptNet triples:

1. Prepare text sentences and the masked token indices at each instance:	

   ```bash
   python prepare_cpnet_finetune.py -cpnet ../ConceptNet/result/retrieval.json\
   -output_dir ../finetune_data/ -add_sep
   ```
   The generated files are `cpnet.txt` for natural language text and `cpnet_mask.txt` for masked tokens in each instance. 

2. Then fine-tune a BERT model using `../run_cpnet_lm.py`:

   ```bash
   python run_cpnet_lm.py --model_type bert --model_name_or_path bert-base-uncased --do_train\
   --train_data_file finetune_data/cpnet.txt --train_pos_file finetune_data/cpnet_mask.txt --do_eval\
   --eval_data_file finetune_data/cpnet.txt --eval_pos_file finetune_data/cpnet_mask.txt\
   --mlm --mlm_probability 1.0 --line_by_line --num_train_epochs 1\
   --warmup_steps 1100 --logging_steps 1100 --save_steps 1100 --eval_all_checkpoints\
   --per_gpu_train_batch_size 32 --gradient_accumulation_steps 1 --per_gpu_eval_batch_size 32\
   --learning_rate 3e-5 --block_size 35 --output_dir OUTPUT_DIR
   ```

   
