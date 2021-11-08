Title: Real-time 2D Detection (waymo open dataset)
Team member: Yingfei Fan(yf2549)


Goal:
Given a set of camera images, produce a set of 2D boxes for the objects in the scene. Model should run faster than 70 ms/frame on a Nvidia Tesla V100 GPU. In the first phase of the project, use only one GUP. In the second phase, use more GPUs (2 or 3) to facilitate the training.

Challenges:
Consider speed and accuracy at the same time.
Use advanced techniques learned from this class such as distributed learning to facilitate the training process.
I don’t have experience with object detection and I really want to get my hands dirty. 

Approach/Techniques:
Models and Frameworks:
Faster RCNN https://arxiv.org/abs/1506.01497
Yolact-resnet50-fpn-pytorch https://docs.openvino.ai/latest/omz_models_model_yolact_resnet50_fpn_pytorch.html
https://arxiv.org/abs/1904.02689
https://arxiv.org/pdf/1612.03144.pdf
Efficient Multiscale learning https://proceedings.neurips.cc/paper/2018/file/166cee72e93a992007a89b39eb29628b-Paper.pdf
Distributed learning with more GPU.

Metrics of accuracy: Average Precision (AP): ∫p(r)dr, where p(r)is the PR curve

Implementation details:
1 ~ 3 Nvidia Tesla V100 GPU (google cloud)
Pytorch
Reusable code: yes there are example code for the models mentioned above
Waymo open data 

Reference:
https://waymo.com/open/challenges/2021/real-time-2d-prediction/

