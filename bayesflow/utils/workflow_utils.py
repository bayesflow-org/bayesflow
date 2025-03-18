import bayesflow.networks
from bayesflow.networks import InferenceNetwork, PointInferenceNetwork, SummaryNetwork


def find_inference_network(inference_network: InferenceNetwork | str, **kwargs) -> InferenceNetwork:
    if isinstance(inference_network, InferenceNetwork) or isinstance(inference_network, PointInferenceNetwork):
        return inference_network
    if isinstance(inference_network, type):
        return inference_network(**kwargs)

    match inference_network.lower():
        case "coupling_flow":
            return bayesflow.networks.CouplingFlow(**kwargs)
        case "flow_matching":
            return bayesflow.networks.FlowMatching(**kwargs)
        case "consistency_model":
            return bayesflow.networks.ConsistencyModel(**kwargs)
        case str() as unknown_network:
            raise ValueError(f"Unknown inference network: '{unknown_network}'")
        case other:
            raise TypeError(f"Unknown transform type: {other}")


def find_summary_network(summary_network: SummaryNetwork | str, **kwargs) -> SummaryNetwork:
    if isinstance(summary_network, SummaryNetwork):
        return summary_network
    if isinstance(summary_network, type):
        return summary_network(**kwargs)

    match summary_network.lower():
        case "deep_set":
            return bayesflow.networks.DeepSet(**kwargs)
        case "set_transformer":
            return bayesflow.networks.SetTransformer(**kwargs)
        case "fusion_transformer":
            return bayesflow.networks.FusionTransformer(**kwargs)
        case "time_series_transformer":
            return bayesflow.networks.TimeSeriesTransformer(**kwargs)
        case "time_series_network":
            return bayesflow.networks.LSTNet(**kwargs)
        case str() as unknown_network:
            raise ValueError(f"Unknown summary network: '{unknown_network}'")
        case other:
            raise TypeError(f"Unknown transform type: {other}")
