from cdt.metrics import SHD, SID_CPDAG


def calculate_precision_recall(true_graph, pred_graph):
    # precision over directed edges, i.e., precision = correct directed edges / all predicted directed edges
    # recall over directed edges, i.e., recall = correct directed edges / all true directed edges
    true_edges = list(true_graph.edges)
    pred_edges = list(pred_graph.edges)
    # remove undirected edges
    true_edges = [edge for edge in true_edges if (edge[1], edge[0]) not in true_edges]
    pred_edges = [edge for edge in pred_edges if (edge[1], edge[0]) not in pred_edges]
    true_positives = len([edge for edge in pred_edges if edge in true_edges])
    all_positive_edges = len(pred_edges)
    all_true_edges = len(true_edges)
    precision = true_positives / all_positive_edges if all_positive_edges > 0 else 0
    recall = true_positives / all_true_edges if all_true_edges > 0 else 0
    return precision, recall

def evaluate(true_graph, pred_graph, skip_sid=False):
    shd = SHD(true_graph, pred_graph, double_for_anticausal=False)
    shd_double_for_anticausal = SHD(true_graph, pred_graph, double_for_anticausal=True)
    if not skip_sid:
        sid_cpdag = SID_CPDAG(true_graph, pred_graph)
        sid_lower = sid_cpdag[0].item()
        sid_upper = sid_cpdag[1].item()
    else:
        sid_lower = 0
        sid_upper = 0
    precision, recall = calculate_precision_recall(true_graph, pred_graph)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return [float(shd), float(shd_double_for_anticausal), float(sid_lower), float(sid_upper), float(precision), float(recall), float(f1)]