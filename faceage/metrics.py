import math
import statistics

AGE_GROUPS = ('0-12', '13-19', '20-39', '40-59', '60+')


def age_group(age):
    if not math.isfinite(float(age)) or not 0 <= float(age) <= 116:
        raise ValueError('Label usia harus berada pada rentang 0-116.')
    for boundary, label in [(12,'0-12'),(19,'13-19'),(39,'20-39'),(59,'40-59')]:
        if age <= boundary:
            return label
    return '60+'


def regression_metrics(targets, predictions):
    targets, predictions = list(targets), list(predictions)
    if not targets or len(targets) != len(predictions):
        raise ValueError('Label dan prediksi harus berjumlah sama dan tidak kosong.')
    if any(not math.isfinite(float(v)) for v in targets + predictions):
        raise ValueError('Label dan prediksi harus bernilai finite.')
    errors = [abs(float(a)-float(b)) for a,b in zip(targets,predictions)]
    return {'count':len(errors), 'mae':statistics.mean(errors),
            'rmse':math.sqrt(statistics.mean(e*e for e in errors)),
            'median_absolute_error':statistics.median(errors)}


def age_group_metrics(targets, predictions):
    targets, predictions = list(targets), list(predictions)
    regression_metrics(targets, predictions)
    result = {}
    for group in AGE_GROUPS:
        pairs = [(a,b) for a,b in zip(targets,predictions) if age_group(a)==group]
        result[group] = regression_metrics(*zip(*pairs)) if pairs else {
            'count':0, 'mae':None, 'rmse':None, 'median_absolute_error':None}
    return result
