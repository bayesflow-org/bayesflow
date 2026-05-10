# Stages

To keep track of the phase each functionality is called in, we provide a `stage` parameter.
There are three stages:

- `training`: The stage to train approximator (and related stateful objects, like the adapter)
- `validation`: Identical setting to `training`, but calls in this stage should _not_ change the approximator
- `inference`: Calls in this change should not change the approximator. In addition, the input structure might be different compared to the training phase. For example for sampling, we only provide `summary_conditions` and `inference_conditions`, but not the `inference_variables`, which we want to infer.
