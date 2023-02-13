"""
## 4 Genetic Programming for Image Classification [20 marks]

Image classification is an important and fundamental task of assigning images to one of the pre-defined groups.
Image classification has a wide range of applications in many domains, including bioinformatics, facial recognition,
remote sensing, and healthcare. To achieve highly accurate image classification, a range of global and local features
must be extracted from any image to be classified. The extracted features are further utilised to build a classifier
that is expected to assign the correct class labels to every image.

In this question, you are provided with two image datasets, namely FEI 1 and FEI 2 1. The two datasets contain many
benchmark images for facial expression classification. All images contain human faces with varied facial expressions.
They are organised into two separate sets of images, one for training and one for testing. Your task is to build an
image classifier for each dataset that can accurately classify any image into two different classes, i.e.,
“Smile” and “Neutral”. There are two steps that you need to perform to achieve this goal, as described in subsequent
subsections.

### 4.1 Automatic Feature Extraction through GP

In this subsection, we use the GP algorithm (i.e., FLGP) introduced in the lectures to design image feature
extractors automatically. You will use the provided strongly-typed GP code in Python to automatically learn suitable
images features respectively for the FEI 1 and FEI 2 datasets, identify the best feature extractors evolved by GP for
both datasets and interpret why the evolved feature extractors can extract useful features for facial expression
classification. Based on the evolved feature extractors, create two pattern files: one contains training examples and
one contains test (unseen) examples, for both the FEI 1 and FEI 2 datasets.

Every image example is associated with one instance vector in the pattern files. Each instance vector has two parts:
the input part which contains the value of the extracted features for the image; and the output part which is the
class label (“Smile” or “Neutral”). The class label can be simply a number (e.g., 0 or 1) in the pattern files.
Choose an appropriate format (ARFF, Data/Name, CSV, etc) for you pattern files. Comma Separated Values (CSV) format
is a good choice; it can easily be converted to other formats. Include a compressed version of your generated data
sets in your submission.

### 4.2 Image Classification Using Features Extracted by GP

Train an image classifier of your choice (e.g., Linear SVM or Na¨ıve Bayes classifier) using the training data and
test its performance on the unseen test data that are obtained from the previous step (Subsection 4.1). Choose
appropriate evaluation criteria (such as classification accuracy) to measure the performance of the trained
classifier on both training and test data. Present and discuss the evaluation results.

Study the best GP trees obtained by you from the previous step (Subsection 4.1), with respect to both the FEI 1 and
FEI 2 datasets. Identify and briefly describe all the global and local image features that can be extract by the GP
trees from the images to be classified. Explain why the extracted global and local image features can enable the
image classifier to achieve good classification accuracy.
"""
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

import time

import pygraphviz

from common import utils

path = utils.get_dataset_path("FEI-dataset")
print("Dataset path: ", path)

import warnings

from sklearn.exceptions import ConvergenceWarning

from IDGP.IDGP_main import *

warnings.filterwarnings(action='ignore', category=ConvergenceWarning, module='sklearn')

# parameters:
population = 100
generation = 30
cxProb = 0.5
mutProb = 0.5
elitismProb = 0.05
initialMinDepth = 2
initialMaxDepth = 6
maxDepth = 8

if __name__ == "__main__":
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
    datasets = os.listdir(path)
    datasets = [str(x).split("_")[0] for x in datasets if os.path.isdir(os.path.join(path, x))]
    datasets = sorted(set(datasets))
    print(datasets)

    random.seed(0)
    np.random.seed(0)

    setParameters(population=population, generation=generation, cxProb=cxProb, mutProb=mutProb,
                  elitismProb=elitismProb, initialMinDepth=initialMinDepth, initialMaxDepth=initialMaxDepth,
                  maxDepth=maxDepth)

    for dataSetName in datasets:
        X_train = np.load(os.path.join(path, dataSetName, dataSetName + '_train_data.npy')) / 255.0
        y_train = np.load(os.path.join(path, dataSetName, dataSetName + '_train_label.npy'))
        X_test = np.load(os.path.join(path, dataSetName, dataSetName + '_test_data.npy')) / 255.0
        y_test = np.load(os.path.join(path, dataSetName, dataSetName + '_test_label.npy'))


        beginTime = time.process_time()

        toolbox = build_toolbox(dataSetName, X_train, y_train)

        pop, log, hof = GPMain(0, toolbox)
        endTime = time.process_time()
        trainTime = endTime - beginTime

        test_features = np.concatenate(
            (toolbox.transform(individual=hof[0], x=X_test, y=y_test), y_test.reshape((-1, 1))), axis=1)

        train_features = np.concatenate(
            (toolbox.transform(individual=hof[0], x=X_train, y=y_train), y_train.reshape((-1, 1))), axis=1)
        testResults = evaluate(individual=hof[0], x=X_train, y=y_train, toolbox=toolbox, cv=False, x_test=X_test,
                               y_test=y_test)

        outpath = utils.get_output_path(question_name=os.path.basename(__file__))
        utils.save_features(train_features, question_name=os.path.basename(__file__),
                            file_name=dataSetName + '_train_features')
        utils.save_features(test_features, question_name=os.path.basename(__file__),
                            file_name=dataSetName + '_test_features')
        endTime1 = time.process_time()
        testTime = endTime1 - endTime

        print(hof.items[0], hof[0])

        print('Best individual ', hof[0])
        print('Test results  ', testResults)
        print('Train time  ', trainTime)
        print('Test time  ', testTime)
        print('End')

        # print gp tree
        G = pygraphviz.AGraph(strict=False, directed=True)
        G.node_attr['style'] = 'filled'
        G.add_node(0)
        utils.render(G, [hof], question_name=os.path.basename(__file__), seed=dataSetName, toolbox=toolbox)