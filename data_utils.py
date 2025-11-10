# """
# Data loading utilities for benchmark datasets.
# Ensure answers are in the correct format for each dataset type.
# """

# def load_gsm8k(n_samples=30):
#     """
#     Load GSM8K math word problems.
#     Returns list of dicts with 'question' and 'answer' (numeric string).
    
#     Example format:
#     {
#         'question': 'Janet has 3 eggs. She buys 2 more. How many does she have?',
#         'answer': '5'  # Just the number, no commas
#     }
#     """
#     try:
#         from datasets import load_dataset
#         dataset = load_dataset("gsm8k", "main", split="test")
        
#         samples = []
#         for i, item in enumerate(dataset):
#             if i >= n_samples:
#                 break
            
#             # GSM8K format: answer is like "#### 42" or "#### 1,234"
#             answer_text = item['answer'].split('####')[-1].strip()
            
#             # Remove commas for consistency
#             import re
#             answer_clean = re.sub(r'[,\s]', '', answer_text)
            
#             # Verify it's a number
#             try:
#                 float(answer_clean)
#                 samples.append({
#                     'question': item['question'],
#                     'answer': answer_clean
#                 })
#             except ValueError:
#                 print(f"Warning: Skipping non-numeric answer: {answer_text}")
#                 continue
        
#         print(f"Loaded {len(samples)} GSM8K samples")
#         return samples
    
#     except Exception as e:
#         print(f"Error loading GSM8K: {e}")
#         print("Using dummy data...")
#         return [
#             {'question': 'What is 5 + 3?', 'answer': '8'},
#             {'question': 'What is 10 - 4?', 'answer': '6'},
#         ]


# def load_strategyqa(n_samples=30):
#     """
#     Load StrategyQA yes/no questions.
#     Returns list of dicts with 'question' and 'answer' ('yes' or 'no').
    
#     Example format:
#     {
#         'question': 'Can a penguin fly?',
#         'answer': 'no'
#     }
#     """
#     try:
#         from datasets import load_dataset
        
#         # Try multiple dataset sources
#         dataset = None
#         sources = [
#             ("ChilleD/StrategyQA", "train"),
#             ("tasksource/strategy-qa", "train"),
#         ]
        
#         for source, split in sources:
#             try:
#                 print(f"Trying to load StrategyQA from {source}...")
#                 dataset = load_dataset(source, split=split)
#                 print(f"Successfully loaded from {source}")
#                 break
#             except Exception as e:
#                 print(f"Failed to load from {source}: {e}")
#                 continue
        
#         if dataset is None:
#             raise Exception("All StrategyQA sources failed")
        
#         samples = []
#         for i, item in enumerate(dataset):
#             if i >= n_samples:
#                 break
            
#             # Convert boolean to yes/no
#             # ChilleD dataset uses 'answer' as boolean
#             answer_value = item.get('answer', False)
            
#             if isinstance(answer_value, bool):
#                 answer = 'yes' if answer_value else 'no'
#             elif isinstance(answer_value, str):
#                 answer = answer_value.lower().strip()
#             else:
#                 answer = 'yes' if answer_value else 'no'
            
#             samples.append({
#                 'question': item['question'],
#                 'answer': answer
#             })
        
#         print(f"Loaded {len(samples)} StrategyQA samples")
#         return samples
    
#     except Exception as e:
#         print(f"Error loading StrategyQA: {e}")
#         print("Using dummy data...")
#         return [
#             {'question': 'Can birds fly?', 'answer': 'yes'},
#             {'question': 'Can fish walk?', 'answer': 'no'},
#             {'question': 'Is the sky blue?', 'answer': 'yes'},
#             {'question': 'Can humans breathe underwater without equipment?', 'answer': 'no'},
#         ]


# def load_commonsenseqa(n_samples=30):
#     """
#     Load CommonsenseQA multiple choice questions.
#     Returns list of dicts with 'question' and 'answer' (letter A-E).
    
#     Example format:
#     {
#         'question': 'What do you use to write? (A) pen (B) car (C) house',
#         'answer': 'A'
#     }
#     """
#     try:
#         from datasets import load_dataset
#         dataset = load_dataset("tau/commonsense_qa", split="validation")
        
#         samples = []
#         for i, item in enumerate(dataset):
#             if i >= n_samples:
#                 break
            
#             # Format question with choices
#             question = item['question']
#             choices = item['choices']
            
#             choice_text = " ".join([
#                 f"({label}) {text}" 
#                 for label, text in zip(choices['label'], choices['text'])
#             ])
            
#             full_question = f"{question} {choice_text}"
            
#             samples.append({
#                 'question': full_question,
#                 'answer': item['answerKey']
#             })
        
#         return samples
    
#     except Exception as e:
#         print(f"Error loading CommonsenseQA: {e}")
#         print("Using dummy data...")
#         return [
#             {
#                 'question': 'What is used for writing? (A) pen (B) car (C) house',
#                 'answer': 'A'
#             },
#             {
#                 'question': 'What do you drink? (A) water (B) rock (C) air',
#                 'answer': 'A'
#             },
#         ]


# # Test function
# if __name__ == "__main__":
#     print("Testing data loaders...")
    
#     gsm8k = load_gsm8k(2)
#     print(f"\nGSM8K sample: {gsm8k[0]}")
    
#     strategyqa = load_strategyqa(2)
#     print(f"\nStrategyQA sample: {strategyqa[0]}")
    
#     commonsenseqa = load_commonsenseqa(2)
#     print(f"\nCommonsenseQA sample: {commonsenseqa[0]}")
"""
Data loading utilities for benchmark datasets.
CRITICAL: Ensures answers are in the correct format for comparison.
"""
import re

def load_gsm8k(n_samples=30):
    """
    Load GSM8K math word problems.
    
    Args:
        n_samples: Number of samples to load. Set to None for all data (~1,319 test samples)
    
    Returns:
        List of dicts with 'question' and 'answer' (numeric string, no commas).
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("gsm8k", "main", split="test")
        
        total_available = len(dataset)
        
        samples = []
        for i, item in enumerate(dataset):
            # Stop if we've reached the limit (only check if n_samples is not None)
            if n_samples is not None and i >= n_samples:
                break
            
            # GSM8K format: answer is like "#### 42" or "#### 1,234"
            answer_text = item['answer'].split('####')[-1].strip()
            
            # Remove commas and whitespace for consistency
            answer_clean = re.sub(r'[,\s]', '', answer_text)
            
            # Verify it's a number
            try:
                float(answer_clean)
                samples.append({
                    'question': item['question'],
                    'answer': answer_clean
                })
            except ValueError:
                print(f"Warning: Skipping non-numeric answer: {answer_text}")
                continue
        
        print(f"Loaded {len(samples)} GSM8K samples (out of {total_available} available)")
        return samples
    
    except Exception as e:
        print(f"Error loading GSM8K: {e}")
        print("Using dummy data...")
        return [
            {'question': 'What is 5 + 3?', 'answer': '8'},
            {'question': 'What is 10 - 4?', 'answer': '6'},
        ]


def load_strategyqa(n_samples=30):
    """
    Load StrategyQA yes/no questions.
    
    Args:
        n_samples: Number of samples to load. Set to None for all data (~2,000+ samples)
    
    Returns:
        List of dicts with 'question' and 'answer' ('yes' or 'no').
    """
    try:
        from datasets import load_dataset
        
        # Try multiple sources
        try:
            print("Trying to load StrategyQA from ChilleD/StrategyQA...")
            dataset = load_dataset("ChilleD/StrategyQA", split="train")
            print("Successfully loaded from ChilleD/StrategyQA")
        except:
            print("Trying wics/strategy-qa...")
            dataset = load_dataset("wics/strategy-qa", split="train")
            print("Successfully loaded from wics/strategy-qa")
        
        total_available = len(dataset)
        
        samples = []
        for i, item in enumerate(dataset):
            # Stop if we've reached the limit (only check if n_samples is not None)
            if n_samples is not None and i >= n_samples:
                break
            
            # Handle different possible answer formats
            answer_raw = item.get('answer', item.get('label', None))
            
            # Convert to yes/no string
            if isinstance(answer_raw, bool):
                answer = 'yes' if answer_raw else 'no'
            elif isinstance(answer_raw, str):
                answer = answer_raw.lower().strip()
                if answer in ['true', '1', 'yes']:
                    answer = 'yes'
                elif answer in ['false', '0', 'no']:
                    answer = 'no'
            else:
                print(f"Warning: Unexpected answer format: {answer_raw}")
                continue
            
            if answer in ['yes', 'no']:
                samples.append({
                    'question': item['question'],
                    'answer': answer
                })
        
        print(f"Loaded {len(samples)} StrategyQA samples (out of {total_available} available)")
        return samples
    
    except Exception as e:
        print(f"Error loading StrategyQA: {e}")
        print("Using dummy data...")
        return [
            {'question': 'Can birds fly?', 'answer': 'yes'},
            {'question': 'Can fish walk on land?', 'answer': 'no'},
        ]


def load_commonsenseqa(n_samples=30):
    """
    Load CommonsenseQA multiple choice questions.
    
    Args:
        n_samples: Number of samples to load. Set to None for all data (~1,221 validation samples)
    
    Returns:
        List of dicts with 'question' and 'answer' (letter A-E).
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("tau/commonsense_qa", split="validation")
        
        total_available = len(dataset)
        
        samples = []
        for i, item in enumerate(dataset):
            # Stop if we've reached the limit (only check if n_samples is not None)
            if n_samples is not None and i >= n_samples:
                break
            
            # Format question with choices
            question = item['question']
            choices = item['choices']
            
            choice_text = " ".join([
                f"({label}) {text}" 
                for label, text in zip(choices['label'], choices['text'])
            ])
            
            full_question = f"{question} {choice_text}"
            
            # Ensure answer is uppercase single letter
            answer = item['answerKey'].strip().upper()
            
            if answer in ['A', 'B', 'C', 'D', 'E']:
                samples.append({
                    'question': full_question,
                    'answer': answer
                })
        
        print(f"Loaded {len(samples)} CommonsenseQA samples (out of {total_available} available)")
        return samples
    
    except Exception as e:
        print(f"Error loading CommonsenseQA: {e}")
        print("Using dummy data...")
        return [
            {
                'question': 'What is used for writing? (A) pen (B) car (C) house',
                'answer': 'A'
            },
            {
                'question': 'What do you drink? (A) water (B) rock (C) air',
                'answer': 'A'
            },
        ]


# Test function
if __name__ == "__main__":
    print("Testing data loaders...")
    
    gsm8k = load_gsm8k(2)
    print(f"\nGSM8K sample:")
    print(f"  Q: {gsm8k[0]['question'][:60]}...")
    print(f"  A: '{gsm8k[0]['answer']}' (type: {type(gsm8k[0]['answer']).__name__})")
    
    strategyqa = load_strategyqa(2)
    print(f"\nStrategyQA sample:")
    print(f"  Q: {strategyqa[0]['question'][:60]}...")
    print(f"  A: '{strategyqa[0]['answer']}' (type: {type(strategyqa[0]['answer']).__name__})")
    
    commonsenseqa = load_commonsenseqa(2)
    print(f"\nCommonsenseQA sample:")
    print(f"  Q: {commonsenseqa[0]['question'][:60]}...")
    print(f"  A: '{commonsenseqa[0]['answer']}' (type: {type(commonsenseqa[0]['answer']).__name__})")