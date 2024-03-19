# Code Implementation for CSQE

## Python Env


```
conda create -n csqe python=3.9
conda activate csqe

conda install -c conda-forge faiss-gpu openjdk=11 maven
pip install transformers==4.20.0
pip install openai pandas matplotlib gpustat ipython jupyter datasets accelerate sklearn tensorboard nltk pytrec_eval prettytable gradio

pip install pyserini
pip install setuptools==59.5.0
python -m spacy download en_core_web_sm
```


## Running Script

### TREC 2019

```
python3 csqe.py \
  --expansion_method keqe/csqe \
  --threads 16 --batch-size 128 --index msmarco-v1-passage \
  --topics dl19-passage --answer_key contents \
  --bm25 --disable_bm25_param --max_demo_len 128 \
  --openai_api_key <API_KEY> --trec_python_path <PYTHON_PATH> \
  --output_dir <OUTPUT_DIR>
```


### TREC 2020

```
python3 csqe.py \
  --expansion_method keqe/csqe \
  --threads 16 --batch-size 128 --index msmarco-v1-passage \
  --topics dl20 --qrels dl20-passage --answer_key contents \
  --bm25 --disable_bm25_param --max_demo_len 128 \
  --openai_api_key <API_KEY> --trec_python_path <PYTHON_PATH> \
  --output_dir <OUTPUT_DIR>
```

### BEIR

```
DATASET="trec-covid"
# can also be "scifact", "arguana", "fiqa", "dbpedia-entity", "trec-news"

python3 csqe.py \
    --expansion_method keqe/csqe \
    --threads 16 --batch-size 128 --index beir-v1.0.0-${DATASET}.flat --topics beir-v1.0.0-${DATASET}-test \
    --answer_key "title|text" --metric_beir \
    --bm25 --disable_bm25_param --max_demo_len 128 \
    --openai_api_key <API_KEY> --trec_python_path <API_KEY> \
    --output_dir <OUTPUT_DIR>
```

When using csqe, please add `--gen_num 2` to constrain both keqe and corpus-originated expansion only generate two generations.

For "arguana", please add `--remove-query`.





