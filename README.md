# Implementation of people detection by using fine tuning and transfer learning

This repo is a fork of the MMDetection repo: https://github.com/open-mmlab/mmdetection.

In this repo we have extended the Cascade R-CNN model by fine tuning it into only detecting one class; namely the person class.

We have also modified the framework in order to use transfer learning, by forcing all other layers than the fully connected ones to be frozen during training.

## Authors:
[Johannes Ernstsen](https://github.com/Ernstsen), [Morten Hansen](https://github.com/MortenErfurt) & [Mathias Jensen](https://github.com/m-atlantis)
