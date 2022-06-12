# Introduction
The work of this library attempt to improve deep learning models predictions and explanations by bringing the explanation correction into the training loop. The existing methodology focuses on producing explanations to increase the model transparency and interpretability not on using explanations as an input. We achieve this goal by constraining explanation through defining an explanation loss function. In the end, we will introduce an algorithm that combines loss on the label and explanation to push to ResNet-18 performing better predictions and better explanations. We illustrate the benefits and possibilities of our approach on various datasets. Furthermore, we monitor the improvement of the prediction
and production of the proper explanation on more challenging tasks such as classification in the presence of confound. Lastly we test test our algorithm and loss on the explanations in a real-world classification scenario on an Italian COVID-19 Lung Ultrasound Image Database (ICLUS-DB) and interpret the results.
## Code structure
Directory names and perpose:
- Covid_input_grad: Containes code for applying method to Italian COVID-19 Lung Ultrasound Image Database (ICLUS-DB) and interpret the results.
- mnist_decoy: Containes code for applying method to MNIST dataset and running different experiments.
- mnist_normal experiments using normal MNIST datasets.
- RRRLoss: Code for testing RRRLoss.
- captumloss: Extracting explanation using captum library (deprecated).

## MAIN EXPERIMENTS:
For running experiments of research question 1 and 2. In this part output will be stored in the log folder. MNIST decoy dataset need to be added to data folder. 
```sh
python mnist_decoy/experiment_2_thesis.py
```

For research question 4 Covid dataset needed to be added to data folder.(Dataset size is around 37GB).
After running the experiments results are stored in the logs folder with tensorboard fileformat. In order to run the result you need to install tensorboard and point it to logs directory.
```sh
python Covid_input_grad/experiments.py
```
Applying co-training algorithm to ICLUS-DB  to prove explanation in the training loop can correct the explanations and improve the accuracy.
```sh
python Covid_input_grad/experiments_colearning.py
```
