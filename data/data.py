import pandas as pd


def load_lucas():
    data_lung_cancer = pd.read_csv("data/lucas0/lucas0_train.targets", header=None)
    data = pd.read_csv("data/lucas0/lucas0_train.data", sep=" ", header=None)
    data = pd.concat([data_lung_cancer, data], axis=1)
    variables = [
        "Lung_Cancer",
        "Smoking",
        "Yellow_Fingers",
        "Anxiety",
        "Peer_Pressure",
        "Genetics",
        "Attention_Disorder",
        "Born_an_Even_Day",
        "Car_Accident",
        "Fatigue",
        "Allergy",
        "Coughing"
    ]
    var_labels = variables
    tags = None
    edges = []
    edges.append(("Lung_Cancer", "Coughing"))
    edges.append(("Lung_Cancer", "Fatigue"))
    edges.append(("Smoking", "Yellow_Fingers"))
    edges.append(("Smoking", "Lung_Cancer"))
    edges.append(("Anxiety", "Smoking"))
    edges.append(("Peer_Pressure", "Smoking"))
    edges.append(("Genetics", "Lung_Cancer"))
    edges.append(("Genetics", "Attention_Disorder"))
    edges.append(("Attention_Disorder", "Car_Accident"))
    edges.append(("Fatigue", "Car_Accident"))
    edges.append(("Allergy", "Coughing"))
    edges.append(("Coughing", "Fatigue"))
    edges.append
    positions = None
    data = data.iloc[:, :-1].to_numpy()
    return variables, var_labels, tags, edges, positions, data