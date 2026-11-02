from datasets import load_dataset
from transformers import AlbertTokenizer, AlbertTokenizerFast

def load_task_data(task, tokenizer, max_seq_length, config):
    if task in ['cola', 'sst2']:
        dataset = load_dataset('glue', task)
        eval_split = 'validation'
        label_column_name = 'label'
        
        def tokenize_function(examples):
            tokenized = tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=max_seq_length)
            tokenized['labels'] = examples[label_column_name]
            return tokenized
    
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset.set_format('torch')
        return tokenized_dataset['train'], tokenized_dataset[eval_split]
    
    elif task == 'sts':
        dataset = load_dataset('glue', 'stsb')
        eval_split = 'validation'
        label_column_name = 'label'
        
        def tokenize_function(examples):
            tokenized = tokenizer(examples['sentence1'], examples['sentence2'], padding='max_length', truncation=True, max_length=max_seq_length)
            tokenized['labels'] = examples[label_column_name]
            return tokenized
    
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset.set_format('torch')
        return tokenized_dataset['train'], tokenized_dataset[eval_split]
    
    elif task in ['mnli', 'qnli', 'qqp', 'rte', 'mrpc', 'wnli']:
        dataset = load_dataset('glue', task)
        eval_split = 'validation_matched' if task == 'mnli' else 'validation'
        label_column_name = 'label'
        
        def tokenize_function(examples):
            sentence1_key = 'premise' if task == 'mnli' else ('question1' if task == 'qqp' else 'sentence1' if task in ['mrpc', 'rte', 'wnli'] else 'question' if task == 'qnli' else 'sentence1')
            sentence2_key = 'hypothesis' if task == 'mnli' else ('question2' if task == 'qqp' else 'sentence2' if task in ['mrpc', 'rte', 'wnli'] else 'sentence' if task == 'qnli' else 'sentence2')
            tokenized = tokenizer(examples[sentence1_key], examples[sentence2_key], padding='max_length', truncation=True, max_length=max_seq_length)
            tokenized['labels'] = examples[label_column_name]
            return tokenized
    
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset.set_format('torch')
        return tokenized_dataset['train'], tokenized_dataset[eval_split]
    
    elif task == 'squad_v1':
        dataset = load_dataset('squad')
        eval_split = 'validation'
        # SQuAD has its own way of handling labels/answers
        def tokenize_function(examples):
            tokenized_examples = tokenizer(
                examples['question'],
                examples['context'],
                padding='max_length',
                truncation=True,
                max_length=max_seq_length,
                return_offsets_mapping=True,
                return_overflowing_tokens=True,
                stride=config.get('doc_stride', 128) # Add stride for overlapping contexts
            )

            # We need to map the character-level answer spans to token-level answer spans.
            # We also handle the case where the answer is not in the tokenized context due to truncation.
            # We will create a list of features, where each feature corresponds to a tokenized input
            # (which might be an original example or an overflowing part of an original example).
            # This is necessary because one original example can yield multiple tokenized features.
            features = {
                'input_ids': [],
                'attention_mask': [],
                'token_type_ids': [],
                'start_positions': [],
                'end_positions': [],
                'example_id': [] # To link back to original example
            }

            for i in range(len(tokenized_examples['input_ids'])):
                input_ids = tokenized_examples['input_ids'][i]
                attention_mask = tokenized_examples['attention_mask'][i]
                token_type_ids = tokenized_examples['token_type_ids'][i] if 'token_type_ids' in tokenized_examples else [0] * len(input_ids)
                offset = tokenized_examples['offset_mapping'][i]
                sample_idx = tokenized_examples['overflow_to_sample_mapping'][i]
                
                # Get original example details
                example_id = examples['id']
                answers = examples['answers']
                
                # Find the start and end of the context
                sequence_ids = tokenized_examples.sequence_ids(i)
                context_start_token_idx = sequence_ids.index(1) if 1 in sequence_ids else None
                context_end_token_idx = len(sequence_ids) - 1 - sequence_ids[::-1].index(1) if 1 in sequence_ids else None

                start_position = 0
                end_position = 0

                if context_start_token_idx is not None and context_end_token_idx is not None:
                    # Only add start/end positions if there's an answer and it's within the context
                    if answers and answers['answer_start'] and answers['text']:
                        start_char = answers['answer_start'][0]
                        end_char = start_char + len(answers['text'][0])

                        # Find start token
                        token_idx = context_start_token_idx
                        while token_idx <= context_end_token_idx and offset[token_idx][0] <= start_char:
                            token_idx += 1
                        start_position = token_idx - 1

                        # Find end token
                        token_idx = context_end_token_idx
                        while token_idx >= context_start_token_idx and offset[token_idx][1] >= end_char:
                            token_idx -= 1
                        end_position = token_idx + 1

                        # If the answer is not fully contained in the context, set positions to (0,0)
                        if not (offset[start_position][0] <= start_char and offset[end_position][1] >= end_char):
                            start_position = 0
                            end_position = 0
                    else: # No answer for this example
                        start_position = 0
                        end_position = 0

                features['input_ids'].append(input_ids)
                features['attention_mask'].append(attention_mask)
                features['token_type_ids'].append(token_type_ids)
                features['start_positions'].append(start_position)
                features['end_positions'].append(end_position)
                features['example_id'].append(example_id)

            return features
        
        columns_to_remove = [col for col in dataset['train'].column_names if col not in ['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions', 'example_id']]
        tokenized_dataset = dataset.map(tokenize_function, batched=False, remove_columns=columns_to_remove)

        tokenized_dataset.set_format('torch')
        
        return tokenized_dataset['train'], tokenized_dataset[eval_split]
    
    elif task == 'squad_v2':
        dataset = load_dataset('squad_v2')
        eval_split = 'validation'
        # SQuAD has its own way of handling labels/answers
        def tokenize_function(examples):
            return tokenizer(examples['question'], examples['context'], padding='max_length', truncation=True, max_length=max_seq_length)
    
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset.set_format('torch')
        return tokenized_dataset['train'], tokenized_dataset[eval_split]
    
    elif task == 'race':
        dataset = load_dataset('race', 'all')
        eval_split = 'validation'
        
        def tokenize_function(examples):
            # For RACE: concatenate article + question for tokenization
            tokenized = tokenizer([article + " " + question for article, question in zip(examples['article'], examples['question'])],
                                  padding='max_length', truncation=True, max_length=max_seq_length)
            tokenized['labels'] = examples['label']
            return tokenized
    
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset.set_format('torch')
        return tokenized_dataset['train'], tokenized_dataset[eval_split]