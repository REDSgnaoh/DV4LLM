# Detoxification experiment
Follow the steps in our pipeline to get accurate evaluations:

1. **Data Preparation**:
   - Objective: Prepare a dataset consisting of various prompts, tailored specifically for toxicity evaluation.

2. **Fine-tuning**:
   - Objective: Fine-tune a model using data curated through a specific data selection method.
   - Note: The current implementation does NOT support the GPT-3 model due to its dependency on an external API.

3. **Generation**:
   - Objective: Generate 25 responses for each prompt in the dataset using the fine-tuned model.

4. **Evaluation**:
   - Involves three key evaluations:
     - **Toxicity Evaluation**: Refer to `toxicity_evaluation/toxicity_evaluation.py` for details.
     - **Perplexity Evaluation**: Further details can be found at [HuggingFace Spaces](https://huggingface.co/spaces/evaluate-metric/perplexity).
     - **Utility Evaluation**: Further details can be found at [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).


