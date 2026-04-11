from csv import reader
from pathlib import Path


def precision_k(reference: str|Path, predict: str|Path) -> dict:
    """
    Calculates precision@k metric for predictions csv file

    Args:
        reference (str|Path): path to reference csv file
        predict (str|Path): path to predict csv file

    Returns:
        dict: dictionary with keys `score` (common score) and `detail` (detail info for every record)
    """
    y = {}
    with open(reference, encoding="utf-8") as reference_file:
        reference_file.readline()
        for line in reader(reference_file):
            y[line[0]] = set(line[1].split(","))

    result = {}
    total_score = 0
    count = 0

    with open(predict, encoding="utf-8") as predict_file:
        predict_file.readline()
        for line in reader(predict_file):
            predictions = set(line[1].split(","))
            result[line[0]] = {
                "extra": list(predictions - y[line[0]]),
                "missing": list(y[line[0]] - predictions),
                "score": 1 - len(predictions - y[line[0]])/len(predictions)
            }
            count += 1
            total_score += result[line[0]]["score"]

    return {
        "score": total_score / count,
        "detail": result
    }