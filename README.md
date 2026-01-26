# general info on how to run dependency parsers and interpreting the benchmark data for each model card
All released public models from bezoku labs are posted in this repo, unless the weights exceed 25 MB.

For Czech and any other files that exceed the github limit, email ian.gilmour@bezoku.ai for a copy.

openVINO runners and related files are work in progress. If you cannot get a model to run using the tools provided, please contact us for support.

Minimum requirements for inference:
torch>=2.0.0
openvino>=2023.0.0
numpy>=1.24.0

The up to date list for languages can be found here - https://github.com/bezokurepo/language-list

# How to understand benchmarks in model cards
UPOS Accuracy: How well the model predicts Universal Part of Speech tags (e.g., NOUN, VERB) which are the most basic morphological category prediction.
XPOS Accuracy: How well the model predicts language specific Part of Speech morphology. This is harder for languages which are inflected, agglutinative etc due to their complexity and the requirement for enough (gold standard) annotated data for the model to learn from. 
DEPREL Accuracy: How well the model predicts each token for a given dependency relationship label. This syntactic metric depends on the HEAD being correctly predicted
FEATS Accuracy: How well the model predicts the syntactic role of each morpheme. This relates to features such as tenses, gender and numbers, for example, and is crucial for downstream semnatic performance.   
HEAD UAS: How well the model predicts tokens that are assigned the correct head (parent) node. If the HEAD prediction is not performing well, by definition the dependency parser will not function effectively.
LAS: (Labeled Attachment Score): How well the model predicts tokens that are assigned the correct HEAD AND the correct dependency label.Because LAS measures accuracy for the sentence HEAD and the dependency labl, it is generally lower than UAS.

# Learn more about models
You can visit our website here for more information on modeling and how to onboard -> https://www.bezoku.tech/service-01
# deprecation
Regular maintenace and updating will be in operation. Deprecation rules are work in progress, keep in touch if you are forking any models
# feedback
If you need support in training our low resource and indigenous language dependency parser for specific tasks (for example, Sentiment Analysis, Grammar Checker, Spell Checker, Domain specific Question & Answer, Document and Internet Search), or deployment with openVINO (infrence via an API or on device) visit https://www.bezoku.tech/service-01
# training data
You can find the original conllu files here -> https://github.com/UniversalDependencies, many of which have already been forked to bezokurepo. 
See the general license.txt and license.txt for each model, which may vary depending on separate licensing / attribution for each corpus. We do our best to keep licensing accurate includng any references. If you spot any mistakes in attribution report them to ian.gilmour@bezoku.ai
#roadmap
Our roadmap for annotatng data can be found here -> https://github.com/bezokurepo/data
