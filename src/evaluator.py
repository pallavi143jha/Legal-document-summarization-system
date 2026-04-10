def rouge_scores(prediction, reference):
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        sc = scorer.score(reference, prediction)
        return {
            "rouge1": round(sc["rouge1"].fmeasure, 4),
            "rouge2": round(sc["rouge2"].fmeasure, 4),
            "rougeL": round(sc["rougeL"].fmeasure, 4),
        }
    except Exception as e:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "error": str(e)}


def interpret_rouge(scores):
    r1 = scores.get("rouge1", 0)
    if r1 >= 0.55:
        return "Excellent"
    elif r1 >= 0.40:
        return "Good"
    elif r1 >= 0.25:
        return "Moderate"
    return "Low"