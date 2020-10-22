# Assembly 6D pose Estimation Dataset demo code

This repository provides code that demonstrates how to use the Assembly 6D Pose Estimation dataset. The dataset contains RGB-D images with ground truth annotations.

The dataset is collected for the project published in IEEE Robotics and Automation Letters (Volume: 5, Issue: 2, Apr. 2020). For more information about the dataset, please refer to the publication ``Learning to Assemble: Estimating 6D Poses for Robotic Object-Object Manipulation''.

The main purpose of this code is to demonstrate how to use the dataset. The dataset is separated into four tasks. Each task contains two synthetically generated training datasets and a test dataset with real images. For each task, we provide polygonal meshes for the example object, the manipulation object, and the template mesh. For each dataset sample, we provide annotations on ground truth poses. The pose shows where the manipulation object should be placed. In the training dataset, we provide segmentation maps for the template mesh, the target object, and distracting objects.


## Running demo code

To run the demonstration code, add a path to the dataset in run_dataset_examples.sh (substitute TODO:DATASET_PATH with the actual path to the dataset). Run the code with the following command:

```
sh run_dataset_examples.sh
```

The codebase uses python 3. The code will sequential plot images from the dataset with ground truth annotations. To get the next image, you need to close the plots that the code will open. 


## Arguments

When running the code, you can pass several arguments. Were we Describe the arguments and options that can be passed as the argument value.

--dataset_root	Path to the folder where the dataset is located.

--task_name		Name of the task you want to visualize. Options: task_1, task_2, task_3, task_4.

--dataset		The dataset variable is used to choose a dataset type. Options: train_original_mesh, train_random_mesh, test. We provide two training datasets. The first one uses the example object mesh to generate the data, while the second uses the shape randomization approach when generating the data. Both training datasets are synthetic. To show test images choose the test option. 

--mesh_name		Name of the mesh to plot. Options: manipulation_object, target_object, template_mesh. When plotting meshes, one can choose to plot the manipulation object, the example target object or the template mesh. These meshes are provided as part of the dataset. In task 3, we added an additional mesh to plot images, as shown in the paper. To plot this additional  mesh, use the option: manipulation_object_paper_plot. However, keep in mind that this is not the polygonal mesh. 

--plot_mesh		Plot the object mesh or bounding box. Options: mesh, bb. The first options plots mesh projection over the image, while bb plots the template bounding box over the image. You can use the bounding box code as an example to obtain the bounding box corners, which are used to train the neural networks described in the paper. 


## Citation

If you use the dataset in your research, please cite the ``Learning to Assemble: Estimating 6D Poses for Robotic Object-Object Manipulation'' paper.

```
@ARTICLE{stevsic2020ral,
author={{Stevšiæ}, Stefan and {Christen}, Sammy and {Hilliges}, Otmar},
journal={IEEE Robotics and Automation Letters},
title={Learning to Assemble: Estimating 6D Poses for Robotic Object-Object Manipulation},
year={2020},
volume={5},
number={2},
pages={1159-1166},
keywords={Deep learning in robotics and automation;perception for grasping and manipulation;computer vision for automation},
doi={10.1109/LRA.2020.2967325},
ISSN={2377-3774},
month={April},}
```
