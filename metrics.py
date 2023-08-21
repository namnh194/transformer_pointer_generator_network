import datasets
from inference import translate_sentence


rouge = datasets.load_metric("rouge")
def rouge_score(valid_src_data, valid_trg_data, model, SRC, TRG, device, k, max_strlen):
    pred_sents = []
    for sentence in valid_src_data:
        pred_trg = translate_sentence(sentence, model, SRC, TRG, device, k, max_strlen)
        pred_sents.append(pred_trg)

    pred_sents = [sent.split() for sent in pred_sents]
    trg_sents = [sent.split() for sent in valid_trg_data]
    rouge_output = rouge.compute(predictions=pred_sents, references=trg_sents, rouge_types=["rouge2"])["rouge2"].mid
    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4)}