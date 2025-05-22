This code is for processing the GenImage dataset provided from https://github.com/GenImage-Dataset/GenImage (downloaded from the google drive link given on the github page) into a multi-task setup for CL. 
	- If you just want one generator you can just use the code for that single generator instead with some minor changes.

I have also uploaded the final processed tensors for all tasks to Google Drive for reproducibility and to skip the need for this process. The overall size is ~60 GB. The link is provided in the Github ReadMe. This code is currently set up in a slightly clunky way where we first created the GenImage-Disjoint tasks and then retroactively extended them to include all tasks for each generator. We will try and reupload a cleaner version that makes all of those tasks from scratch without assuming the existence of GenImage-Disjoint and any associated intermediate output files/dictionaries.


GenImage provides several generators, where for each generator there is a file structure like:
	-  Train
		- Synthetic Images
		- Natural Images
	- Validation
		- Synthetic Images
		- Natural Images

Note that since everything is based on ImageNet, there is no publically available test set. We use the validation subset as a test set, but if you need a validation split for training purposes you can split one off from the train data.

One big thing I will note up front, and I kind've forget exactly what the issue was, but for some reason the makers of GenImage mixed ImageNet notations. ImageNet assigns string IDs to each label instead of english words or integers. For some reason though for a given class, the ID differs between (I believe) the validation and training data for GenImage, so two different dictionaries were needed to map them back to the same integer label for that class. 
	- I don't recall exactly which dictionary was used where off the top of my head, but it shoud be clear enough from looking at the code provided and cross-referencing the dictionaries and image files in the dataset if needed.
	- As far as running the code is concerned, all of the dictionaries needed are provided so this is mostly so that you understand what they're being used for as this was a big point of confusion when working with GenImage for me.


The files are split into a Scripts and Dictionaries directory for organization, but to run them they all need to be moved into the "Unzipped" folder together unless you adjust the paths in the scripts to reflect the subdirectory layout.

In the Scripts subdirectory, the .slurm file is the entry point for running each step of the processing. The dataset files first have to be downloaded as a set of zipped files. They then need to be unzipped and moved into the "Unzipped" folder for processing.

The processing files are run in the following order through the slurm file (or command line):
1. ProduceSubsetDicts.py:
	- Creates dictonaries that dictate which classes are used for each generator, since this was CL we wanted each task (generator) to use a different 100 classes of ImageNet, with the 200 classes in Tiny ImageNet set aside for pretraining.
	- Note it has both "produceDicts() and updateDicts()" in main(), this and a number of other points of confusion are because I went back and retroactively updated my datasets to have additional tasks, so rather than break the code by removing that haphazard addition I'm going to just point out where it was added so it can be avoided. Here you just want produceDicts() if none already exist.

2.processGeneratorImages.py:
	- Reads from the produced dictionaries to process the images from the unzipped directories and copy over the ones belonging only to the desired subset into a new directory.

3.  processTensors.py:
	- Reads the newly created split directory of images for the given generator and converts them into a pytorch tensor to be compatible with how we loaded our images. Can probably skip this step if you use something like PIL to load the images from the directory directly, but I'm not sure as I always work with tensor inputs.

