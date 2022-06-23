# Star-Lab
The goal is to test the use of machine learning tools to identify "spectral blend binaries" among a sample of SpeX prism spectra. Spectral blend binaries are unresolved binary systems containing a late-M or L dwarf primary plus a T dwarf secondary, for which the combined-light spectrum shows peculiarities related to the different spectral types of the components.

## Week 06/20 Notes:
Working on visualizing data and data augmentation. This is the main issue because single stars are grouped with binaries of the same spectral class. Spectral classes need to be split up and there needs to be more generated single star data with Gaussian Noise. Will need to organize data into training, validation, and testing data. Considering saving 20% of the data for testing, with 80% for training/validation using Cross Validation. I may be able to get away with not having testing data because the real test is how this performs on real world flux values.

Working on setting up a simple baseline of CNN to keep track of accuracies as we improve our data. I am setting up an environment, so that I can quickly edit model architectures and run experiments faster. Need to look into getting a GPU or access to the GPU clusters at UCSD.
