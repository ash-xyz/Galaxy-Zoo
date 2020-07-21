# Galaxy-Zoo
Galaxy Zoo - The Galaxy Challenge

Model 1: Achieved a root mean squared loss of 0.11314

Used a Nasnet Mobile Architecture, No Data Augmentation, 75/25 split for training and validation data, learning rate = 0.001, 25 epochs, Adam optimizer

Model 2: 0.11896

NasnetMobile, Rotations of 180 degrees, vertical and horizontal flip, 90/10 split for training and validation data, learning rate = 0.001, 25 epochs, Adam optimizer

Changed rotation to 90 degrees, MobileNetV2