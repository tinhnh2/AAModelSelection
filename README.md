# AAModelSelection
The amino acid substitution model describing the substation rates among amino acids is a key component in studying the evolutionary relationships among species from protein sequences.  The amino acid models consist of a large number of parameters; therefore, they are estimated from large datasets with hundreds of alignments and used for analyzing individual alignments. A number of general models (e.g., LG, WAG, JTT, Q.pfam) or clade–specific models (e.g., Q.plant, Q.bird, Q.yeast, Q.mammal and Q.insect) have been introduced and widely used in phylogenetic analyses. The first task in analyzing protein sequences is selecting the best appropriate available models for the alignment under study. This can be done by selecting the model with maximum likelihood value that requires much running time for large alignments. Recently, machine learning methods such as ModelRevelator and ModelTeller have been proposed and work well for nucleotide data. that worked well on simulation DNA alignments. In this paper, we propose an efficient method, called ModelExpress to extract features from protein alignments to quickly train a convolution neuron network on a personal computer for selecting amino acid models. Experiments on both real and simulated data showed that ModelExpress performed well on simulation data. It was better than ModelFinder on empirical data from clade genomes.

Some steps to use:
- Step 1: download the source code and datasets by command:\
  git clone
- Step 2: run script to extract information using pairwise script or triplet script.\
  python triplet_extraction.py folder label\
  folder: the path to alignments\
  label: the true label of alignments
- Step 3: run script to predict alignments:
  python predict_model.py alignment.csv\
  alignment.csv: the output of extraction script

