evalml experiment config/forecasters-ich1-stage-C-vs-limited-new.yaml --report -- -R prepare_inference_forecaster 

evalml experiment config/forecasters-ich1-stage-C-vs-subgrid.yaml --report -- -R prepare_inference_forecaster

evalml experiment config/forecasters-ich1-stage-C-encoder-variations-FIXED.yaml --report -- -R prepare_inference_forecaster

evalml experiment config/forecasters-ich1-stage-E-vs-horography.yaml --report -- -R prepare_inference_forecaster

evalml experiment config/forecasters-ich1-stage-E-vs-knn5-dec.yaml --report -- -R prepare_inference_forecaster
