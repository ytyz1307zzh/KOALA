# KOALA
Pytorch implementation of **K**n**O**wledge-**A**ware procedura**L** text underst**A**nding model on ProPara dataset. Built on the previous state-of-the-art model [NCET](aclweb.org/anthology/W19-1502/), the KOALA model integrates commonsense knowledge from ConceptNet and is trained in a multi-stage training schema. The model achieves 70.4 F1 on the test set on ProPara dataset, a benchmark dataset of procedural text understanding.

## Data

KOALA uses the [ProPara dataset](http://data.allenai.org/propara/) proposed by AI2. This dataset is about a reading comprehension task on procedural text, *i.e.*, a text paragraph that describes a natural process (*e.g.*, photosynthesis, evaporation, etc.). AI models are required to read the paragraph, then predict the state changes (CREATE, MOVE, DESTROY or NONE) as well as the locations of the given entities.

<img src="./image/propara.png" alt="image-20191223140902073" width=500 />

AI2 released the dataset [here](https://docs.google.com/spreadsheets/d/1x5Ct8EmQs2hVKOYX7b2nS0AOoQi4iM7H9d9isXRDwgM/edit#gid=832930347) in the form of Google Spreadsheet. We need three files to run the KOALA model, *i.e.*, the Paragraphs file for the raw text, the Train/Dev/Test file for the dataset split, and the State_change_annotations file for the annotated entities and their locations. I also provide a copy in `data/` directory which is identical to the official release.

**P.S.** Please download the files in CSV format.

## Setup

1. Create a virtual environment with python >= 3.7.

2. Install the dependency packages in `requirements.txt` using `setup.py`:

   ```bash
   pip install .
   ```

3. If you want to create your own dataset using `preprocess.py`, you also need to download the en_core_web_sm model for English language support of SpaCy:

   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

1. [Download](https://docs.google.com/spreadsheets/d/1x5Ct8EmQs2hVKOYX7b2nS0AOoQi4iM7H9d9isXRDwgM/edit#gid=832930347) the dataset or use my copy in `data/`.

2. Process the CSV data files:

   ```bash
   python preprocess.py
   ```

   By default, the files should be put in `data/` and the output JSON files are also stored in `data/`. You can specify the input and output paths using optional command-line arguments. Please refer to the code for more details of command-line arguments.

   Time for running the pre-process script may vary according to your CPU performance. It takes me about 50 minutes on a Intel Xeon 3.7GHz CPU.

3. Train a KOALA model:

   ```bash
   python train.py -mode train -ckpt_dir ckpt -train_set data/train.json -dev_set data/dev.json\
   -cpnet_path CPNET_PATH -cpnet_plm_path CPNET_PLM_PATH -cpnet_struc_input -state_verb STATE_VERB_PATH\
   -wiki_plm_path WIKI_PLM_PATH -finetune
   ```

   where `-ckpt_dir` denotes the directory where checkpoints will be stored.

   `CPNET_PATH` should point to the retrieved ConceptNet knowledge triples. `CPNET_PLM_PATH` should point to the BERT model pre-fine-tuned on ConCeptNet triples. `STATE_VERB_PATH` should point to the co-appearance verb set of entity states. Please refer to `ConceptNet/` for detail or use my copy (see Reproducibility section below. 

   `WIKI_PLM_PATH` should point to the BERT model pre-fine-tuned on Wiki paragraphs. Please refer to `wiki/` for detail or use my copy (see Reproducibility section below)

   Some useful training arguments:

   ```
   -save_mode     Checkpoint saving mode. 'best' (default): only save the best checkpoint on dev set. 
                  'all': save all checkpoints. 
                  'none': don't save checkpoints.
                  'last': save the last checkpoint.
                  'best-last': save the best and the last checkpoints.
   -epoch         Number of epochs to run the dataset. You can set it to -1 
                  to remove epoch limit and only use early stopping 
                  to stop training.
   -impatience    Early stopping rounds. If the accuracy on dev set does not increase for -impatience rounds, 
                  then stop the training process. You can set it to -1 to disable early stopping 
                  and train for a definite number of epochs.
   -report        The frequency of evaluating on dev set and save checkpoints (per epoch).
   ```

   Time for training a new model may vary according to your GPU performance as well as your training schema (*i.e.*, training epochs and early stopping rounds). It takes me about 1 hour to train a new model on a single Tesla P40.

4. Predict on test set using a trained model:

   ```bash
   python -u train.py -mode test -test_set data/test.json -dummy_test data/dummy-predictions.tsv\
   -output predict/prediction.tsv -cpnet_path CPNET_PATH -cpnet_plm_path CPNET_PLM_PATH\
   -cpnet_struc_input -state_verb STATE_VERB_PATH -wiki_plm_path WIKI_PLM_PATH -restore ckpt/best_checkpoint.pt
   ```

   where -output is a TSV file that will contain the prediction results, and -dummy_test is the output template to simplify output formatting. The `dummy-predictions.tsv` file is provided by the [official evaluation script](https://github.com/allenai/aristo-leaderboard/tree/master/propara/data/test) of AI2, and I just copied it to `data/`.

5. Download the [official evaluation script](https://github.com/allenai/aristo-leaderboard/tree/master/propara) of ProPara provided by AI2.

6. Run the evaluation script using the ground-truth labels and your predictions:

   ```bash
   python evaluator.py -p data/prediction.tsv -a data/answers.tsv --diagnostics data/diagnostic.txt
   ```

   where `answers.tsv` contains the ground-truth labels, and `diagnostic.txt` will contain detailed scores for each instance. `answers.tsv` can be found [here](https://github.com/allenai/aristo-leaderboard/tree/master/propara/data/test), or you can use my copy in `data/`. `evaluator.py` is the evaluation script provided by AI2, and can be found [here](https://github.com/allenai/aristo-leaderboard/tree/master/propara/evaluator).

   **P.S.** You should download the whole repo provided by AI2 instead of only downloading `evaluator.py`.

## Reproducibility

To reproduce the 70.4 result on the ProPara test set, you may:

1. Use my own copy of train/dev/test data in `data/`, instead of downloading a new copy from AI2 and process it with `preprocess.py`. Although running the script would generate almost the same result, I'm sorry to say that I used the Python Set in `preprocess.py` and therefore the generated data will have a different order in **location candidates**. Although this is their only difference, it will lead to slightly different embeddings in the location predictor due to the existence of dropout technique.

2. I have uploaded my retrieved ConceptNet knowledge triples, as well as the co-appearance verb set of entity states in `ConceptNet/result/`.

3. I have uploaded my fine-tuned knowledge encoder on ConceptNet and the fine-tuned text encoder on Wiki paragraphs [here]().

4. Using the aforementioned input files, you will be able to reproduce the results using the following training and testing commands:

   Training:

   ```bash
   python -u train.py -mode train -epoch 20 -impatience -1 -save_mode best\
   -ckpt_dir ckpt -cpnet_plm_path MY_CPNET_PLM_MODEL -cpnet_struc_input\
   -cpnet_path MY_CPNET_PATH -state_verb MY_STATE_VERB_PATH -wiki_plm_path MY_WIKI_PLM_MODEL\
   -finetune -hidden_size 256 -attn_loss 0.5 -loc_loss 0.3 -per_gpu_batch_size 32\
   -lr 3e-5 -dropout 0.4
   ```

   Testing:

   ```bash
   python -u train.py -mode test -dummy_test data/dummy-predictions.tsv\
   -output data/prediction.tsv -cpnet_plm_path MY_CPNET_PLM_MODEL -cpnet_struc_input\
   -cpnet_path MY_CPNET_PATH -state_verb MY_STATE_VERB_PATH -wiki_plm_path MY_WIKI_PLM_MODEL\
   -hidden_size 256 -per_gpu_batch_size 32 -restore ckpt/best_checkpoint.pt
   ```

   

