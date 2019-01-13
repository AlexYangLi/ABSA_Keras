Aspect Based Sentiment Analysis (ABSA)
Task 4 of SemEval 2014
-----------------------------------------------------

This folder contains scripts/code for:

A. Running the ABSA baselines.
B. Evaluating the output of your system.
C. Validating the XML file that you will submit to ABSA 2014.


Running the Baselines
-----------------------

The semeval_base.py script is an implementation of the baselines of SemEval Task 4 (Aspect Based Sentiment Analysis).
A high level description of them can be found at the following address:

http://alt.qcri.org/semeval2014/task4/data/uploads/baselinesystemdescription.pdf

By running python semeval_base.py from you shell, a list of possible options will be displayed.
(**Caution: We tested semeval_base.py only in Linux.)

Assuming that rest.xml and lap.xml are the training data for the restaurants and laptops 
domain, respectively, we recommend you run the baselines as follows:

-- restaurants

python semeval_base.py --train rest.xml --task 5

It reads the given data (rest.xml) and splits them in a train (absa--train.xml) and a test part (absa--test.xml) using a 80:20 ratio.
Then, it tags the sentences of the test part with the found aspect terms and categories and stores the result to absa--test.predicted-stageI.xml.
absa--test.gold.xml contains the gold (correct) aspect terms and categories.


python semeval_base.py --train rest.xml --task 6

It reads the given data (rest.xml) splits them in a train (absa--train.xml) and a test part (absa--test.xml) using a 80:20 ratio.
Then, it finds the polarity for the aspect terms and categories of the test part and stores the result to absa--test.predicted-stageII.xml.
absa--test.gold.xml contains the gold (correct) polarities.

-- laptops

python semeval_base.py --train lap.xml --task 1

It reads the given data (lap.xml), splits them in a train (absa--train.xml) and a test part (absa--test.xml) using a 80:20 ratio.
Then, it tags the sentences of the test part with the found aspect terms and stores the result to absa--test.predicted-aspect.xml.
absa--test.gold.xml contains the gold (correct) aspect terms and categories

python semeval_base.py --train lap.xml --task 3

It reads the given data (rest.xml), splits them in a train (absa--train.xml) and a test part (absa--test.xml) using a 80:20 ratio.
Then, it finds the polarity for the aspect terms of the test part and stores the result to absa--test.predicted-stageII.xml.
absa--test.gold.xml contains the gold (correct) polarities.


In all cases above, the baseline script calculates and displays evaluation scores (precision, recall, and F1 for aspect term and aspect category extraction; accuracy for aspect term and aspect category polarity detection).


Evaluation
-----------------------

java -cp ./eval.jar Main.Aspects test.xml ref.xml

It calculates and displays the precision, recall and F1 for aspect term and category extraction for a system that generated test.xml, comparing it to ref.xml that contains
the gold correct annotations. The same measures are also calculated and displayed by semeval_base.py.

java -cp ./eval.jar Main.Polarity test.xml  ref.xml

In contrast to semeval_base.py that calculates only the overall accuracy for the polarity detection task, the above command also calculates F1, Precision and Recall 
for all labels (positive|negative|neutral|conflict). As previously, test.xml is the file that the system generated and ref.xml
is the one that contains the gold (correct) annotations.


Submit your system
-----------------------

The Aspect Based Sentiment Analysis task will run in two stages. 

In the first stage, you will be provided with a XML file that will contain a set of sentences.
If you want to participate in this stage, you have to return a file tagged with the aspect terms and categories in the same way they are tagged in the training data.  

In the second stage, we will provide you with the correct aspect terms and categories  
and you will have to find their polarity (positive|negative|neutral|conflict) and tag them as in the training data.


Before uploading your results (for stage one or two), we highly recommend you validate (as shown below) the XML your system produced against the provided XSD schema (SemEvalSchema.xsd). 
This way you will verify that your XML output is well-formed and can be processed/parsed by our evaluation scripts.

java -cp ./eval.jar Main.Valid test.xml  SemEvalSchema.xsd
	



