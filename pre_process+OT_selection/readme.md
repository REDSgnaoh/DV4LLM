# Data Selection based on OT gradients

### Data pre-processing and domain similarity tests ('pre-process+domain_tests.ipynb')
- **This notebook uses 'cola' sub-task from GLUE benchmark for demonstration. The process for other tasks are essentially the same.**

**Pre-processing** is first performed for both target task samples and candidate samples where we normalize all samples to a fixed length of 1000 characters to avoid different padding patterns affect the analysis on distributional distances.

- For samples with lengths much shorter than 1000 characters (e.g., training data for 'cola'), we concatenate multiple samples to reach 1000 characters; for samples much longer than 1000 characters (e.g., scientific papers), we split each of the original samples to multiple samples of 1000 characters.

- Then, we tokenize the processed samples using BERT tokenizers and embed the tokens using distilledBERT fine-tuned on the target task.

For **domain similarity tests**, we randomly sample 20k samples from each of the 7 domains in the candidate data ['amz', 'wiki', 'news', 'pubmed', 'arxiv', 'book1', 'owtc']. We then tokenize and embed these samples in the same way.

- To analyze the domain similarity, we compute the OT distance between the embeddings of target task samples and the embeddings of samples from each domain.

- Then, we select 2~4 domains with the smallest OT distances to the target task data and use samples from these domains as the candidate data for further selection. Domains with large OT distances will be discarded for this task.

### Data selection based on OT gradients (ot-selection.ipynb)
- **This notebook uses 'cola' sub-task from GLUE benchmark for demonstration. The process for other tasks are essentially the same.**
Selects samples from processed candidate data by solving the OT problem, completing the selection process.
