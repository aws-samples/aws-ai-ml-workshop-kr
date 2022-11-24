import torch
import numpy as np
from tqdm import tqdm
from transformers.data.processors.squad import SquadV2Processor
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))


def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)


def get_gold_answers(example):
    """helper function that retrieves all possible true answers from a squad2.0 example"""
    
    gold_answers = [answer["text"] for answer in example.answers if answer["text"]]

    # if gold_answers doesn't exist it's because this is a negative example - 
    # the only correct answer is an empty string
    if not gold_answers:
        gold_answers = [""]
        
    return gold_answers


def get_prediction(question, context, tokenizer, model):
    # given a question id (qas_id or qid), load the example, get the model outputs and generate an answer
    #question = examples[qid_to_example_index[qid]].question_text
    #context = examples[qid_to_example_index[qid]].context_text
    inputs = tokenizer.encode_plus(question, context, truncation=True, max_length=384, stride=64, return_tensors='pt').to(device)    
    outputs = model(**inputs)

    answer_start = torch.argmax(outputs['start_logits'], dim=1)  # get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(outputs['end_logits'], dim=1) + 1 

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

    return answer


def get_metrics_korquadv1(valid_dir, tokenizer, model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    processor = SquadV2Processor()
    examples = processor.get_dev_examples(valid_dir, filename="KorQuAD_v1.0_dev.json")

    qid_to_example_index = {example.qas_id: i for i, example in enumerate(examples)}
    qid_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
    answer_qids = [qas_id for qas_id, has_answer in qid_to_has_answer.items() if has_answer]
    no_answer_qids = [qas_id for qas_id, has_answer in qid_to_has_answer.items() if not has_answer] 
    
    em_scores = np.array([])
    f1_scores = np.array([])
    
    for qid in tqdm(range(len(answer_qids))):
        # given a question id (qas_id or qid), load the example
        answer_qid = answer_qids[qid]
        question = examples[qid_to_example_index[answer_qid]].question_text
        context = examples[qid_to_example_index[answer_qid]].context_text
        prediction = get_prediction(question, context, tokenizer, model)
        example = examples[qid_to_example_index[answer_qid]]
        gold_answers = get_gold_answers(example)

        em_score = max((compute_exact_match(prediction, answer)) for answer in gold_answers)
        f1_score = max((compute_f1(prediction, answer)) for answer in gold_answers)
        em_scores = np.append(em_scores, em_score)
        f1_scores = np.append(f1_scores, f1_score)    
    
    em = np.mean(em_scores)
    f1 = np.mean(f1_scores)
    
    return em, f1