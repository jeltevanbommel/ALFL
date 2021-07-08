# Active Learning during Federated Learning for Object Detection
The code for the paper 'Active Learning during Federated Learning for Object Detection', presented at the 35th Twente Student Conference on IT.

## Abstract
Convolutional Neural Networks (CNNs) are currently among the most successful machine-learning techniques for object detection. One weakness of CNNs is that they require many labelled examples in order to train a model. This gives problems when training a model on decentralized data, such as in federated learning, where labels may not be available. Training on decentralized data is preferable, due to the benefits in privacy, and decreases in central data storage.  Active learning can solve the unlabeled data problem, by selecting a portion of the unlabeled data and labelling it with an oracle. This paper explores, implements and evaluates several schemes which use active learning to label images locally and then use federated learning to train a global object detection model. Analysis shows the schemes maintain average precision close to centralized learning for heterogeneous data. A novel approach based on a chain of devices allows for increased precision, while decreasing communication costs. The paper shows feasibility of training object detection models with active and federated learning, bringing the benefits of federated learning to the field of object detection. 

## >> Code is on its way :) <<

### Why isn't it here yet? 😡😡
I currently have about 25 different Python Notebooks and scripts that all do different components of the research. I know what each of them do, but for an outsider without documentation it's horrible. So I need a few days to make one neat codebase out of this, such that you can download and run it with 1 click (e.g. a Run on colabs button). Will also upload all of the files needed to completely reproduce the research here, plus the distribution of these files. 

If you'd like a notification when the code is here, let me know via the email address in the paper 😀