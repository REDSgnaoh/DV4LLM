# GLUE experiment
Follow the steps in our pipeline to get GLUE scores:

1. Train the model with selected datasets with masked language modeling (MLM) training by running the shell file:

	`bash mlm_all_8tasks_ot_dsir_295.sh`.

2. Once we trained the models, we fine tune on each of the GLUE task to get the scores:

	`bash run_glue_ot_select.sh`.
	
