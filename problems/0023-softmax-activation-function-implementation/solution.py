import math

def softmax(scores: list[float]) -> list[float]:
    soft_result = []
    for z in scores:
        soft_result.append(math.exp(z - max(scores)))
    soft_result = [round(x/sum(soft_result), 4) for x in soft_result] 

    return soft_result