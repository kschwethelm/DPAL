# Source: https://opacus.ai/tutorials/building_text_classifier
import os
import pandas as pd

import torch
from torch.utils.data import TensorDataset

import transformers
from transformers.data.processors.utils import InputExample
from transformers.data.processors.glue import glue_convert_examples_to_features

def _create_examples(df, set_type, labels):
    """ Convert raw dataframe to a list of InputExample. Filter malformed examples
    """
    examples = []
    for index, row in df.iterrows():
        if row['gold_label'] not in labels:
            continue
        if not isinstance(row['sentence1'], str) or not isinstance(row['sentence2'], str):
            continue
            
        guid = f"{index}-{set_type}"
        examples.append(
            InputExample(guid=guid, text_a=row['sentence1'], text_b=row['sentence2'], label=row['gold_label']))
    return examples

def _df_to_features(df, set_type, labels, max_seq_length, tokenizer):
    """ Pre-process text. This method will:
    1) tokenize inputs
    2) cut or pad each sequence to max_seq_length
    3) convert tokens into ids
    
    The output will contain:
    `input_ids` - padded token ids sequence
    `attention mask` - mask indicating padded tokens
    `token_type_ids` - mask indicating the split between premise and hypothesis
    `label` - label
    """
    examples = _create_examples(df, set_type, labels)
    
    #backward compatibility with older transformers versions
    legacy_kwards = {}
    from packaging import version
    if version.parse(transformers.__version__) < version.parse("2.9.0"):
        legacy_kwards = {
            "pad_on_left": False,
            "pad_token": tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            "pad_token_segment_id": 0,
        }
    
    return glue_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        label_list=labels,
        max_length=max_seq_length,
        output_mode="classification",
        **legacy_kwards,
    )

def _features_to_dataset(features):
    """ Convert features from `_df_to_features` into a single dataset
    """
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long
    )
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long
    )
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels
    )

    return dataset

def get_slni_dataset(root, tokenizer, split, max_seq_length=128):
    """ Load SNLI dataset and convert it into a dataset
    """
    labels = ["contradiction", "entailment", "neutral"]

    snli_folder = os.path.join(root, "snli_1.0")
    path =  os.path.join(snli_folder, f"snli_1.0_{split}.txt")

    df = pd.read_csv(path, sep='\t')
    features = _df_to_features(df, split, labels, max_seq_length, tokenizer)
    dataset = _features_to_dataset(features)
    return dataset