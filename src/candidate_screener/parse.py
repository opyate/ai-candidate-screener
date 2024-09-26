import yaml


def parse_prediction(prediction):
    prediction_yaml = prediction.split("<match>")[1].split("</match>")[0]
    prediction_yaml_dict = yaml.load(prediction_yaml, Loader=yaml.CLoader)
    return prediction_yaml_dict


def parse_summary(prediction):
    summary = prediction.split("<summary>")[1].split("</summary>")[0]
    return summary.strip()
