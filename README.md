# NeuralNetwork

## Example run

Cat dataset loaded successfully
X_train_cat: (12288, 209)
y_train_cat: (1, 209)
X_test_cat: (12288, 50)
y_test_cat: (1, 50)

Mnist dataset loaded successfully
X_train_mnist: (784, 5000)
y_train_mnist: (10, 5000)
X_test_mnist: (784, 1000)
y_test_mnist: (10, 1000)

Sequential cat dataset test

Epoch: 0 || Train loss: 0.734889 || Test loss: 1.17598 || Train accuracy: 38.2775% || Test accuracy: 34%
Epoch: 10 || Train loss: 0.612785 || Test loss: 0.781068 || Train accuracy: 66.0287% || Test accuracy: 32%
Epoch: 20 || Train loss: 0.576315 || Test loss: 0.737011 || Train accuracy: 68.8995% || Test accuracy: 34%
Epoch: 30 || Train loss: 0.54572 || Test loss: 0.708013 || Train accuracy: 71.2919% || Test accuracy: 42%
Epoch: 40 || Train loss: 0.519914 || Test loss: 0.720245 || Train accuracy: 76.555% || Test accuracy: 42%
Epoch: 50 || Train loss: 0.517557 || Test loss: 0.804646 || Train accuracy: 81.8182% || Test accuracy: 38%
Epoch: 60 || Train loss: 0.512444 || Test loss: 0.860359 || Train accuracy: 80.8612% || Test accuracy: 38%
Epoch: 70 || Train loss: 0.498269 || Test loss: 0.868639 || Train accuracy: 79.9043% || Test accuracy: 38%
Epoch: 80 || Train loss: 0.483243 || Test loss: 0.862378 || Train accuracy: 79.9043% || Test accuracy: 38%
Epoch: 90 || Train loss: 0.469468 || Test loss: 0.856197 || Train accuracy: 80.3828% || Test accuracy: 38%

Sequential cat training time: 111.209 seconds

Truth: 1 || Pred: 0

OpenMP cat dataset test

Epoch: 0 || Train loss: 0.79146 || Test loss: 1.07842 || Train accuracy: 36.8421% || Test accuracy: 34%
Epoch: 50 || Train loss: 0.562205 || Test loss: 0.49846 || Train accuracy: 67.9426% || Test accuracy: 74%
Epoch: 100 || Train loss: 0.449977 || Test loss: 0.52923 || Train accuracy: 74.1627% || Test accuracy: 76%
Epoch: 150 || Train loss: 0.36433 || Test loss: 0.45033 || Train accuracy: 80.8612% || Test accuracy: 82%
Epoch: 200 || Train loss: 0.411648 || Test loss: 0.437191 || Train accuracy: 76.555% || Test accuracy: 84%
Epoch: 250 || Train loss: 0.346902 || Test loss: 0.430631 || Train accuracy: 79.9043% || Test accuracy: 86%

OpenMP cat training time: 23.6073 seconds

Truth: 1 || Pred: 0

CUDA cat dataset test

Epoch: 0 || Train loss: 1.63038 || Test loss: 0.700767 || Train accuracy: 34.4498% || Test accuracy: 58%
Epoch: 100 || Train loss: 0.496548 || Test loss: 0.66069 || Train accuracy: 77.512% || Test accuracy: 54%
Epoch: 200 || Train loss: 0.414483 || Test loss: 0.489289 || Train accuracy: 77.9904% || Test accuracy: 78%
Epoch: 300 || Train loss: 0.32657 || Test loss: 0.462734 || Train accuracy: 82.7751% || Test accuracy: 82%
Epoch: 400 || Train loss: 0.249982 || Test loss: 0.459171 || Train accuracy: 88.5167% || Test accuracy: 82%
Epoch: 500 || Train loss: 0.182396 || Test loss: 0.470276 || Train accuracy: 94.2584% || Test accuracy: 80%
Epoch: 600 || Train loss: 0.122497 || Test loss: 0.494631 || Train accuracy: 98.0861% || Test accuracy: 82%

CUDA cat training time: 3.84925 seconds

Truth: 1 || Pred: 0

Sequential mnist dataset test

Epoch: 0 || Train loss: 2.43224 || Test loss: 2.43875 || Train accuracy: 10.5% || Test accuracy: 8.3%
Epoch: 10 || Train loss: 2.29513 || Test loss: 2.31769 || Train accuracy: 14.8% || Test accuracy: 12%
Epoch: 20 || Train loss: 2.20194 || Test loss: 2.2343 || Train accuracy: 19.86% || Test accuracy: 16.6%
Epoch: 30 || Train loss: 2.11855 || Test loss: 2.15998 || Train accuracy: 26.18% || Test accuracy: 22.4%
Epoch: 40 || Train loss: 2.03654 || Test loss: 2.08782 || Train accuracy: 32.36% || Test accuracy: 27.6%
Epoch: 50 || Train loss: 1.953 || Test loss: 2.01401 || Train accuracy: 38.56% || Test accuracy: 34.7%
Epoch: 60 || Train loss: 1.86573 || Test loss: 1.93611 || Train accuracy: 46.42% || Test accuracy: 42%
Epoch: 70 || Train loss: 1.77435 || Test loss: 1.85389 || Train accuracy: 53.74% || Test accuracy: 47.5%
Epoch: 80 || Train loss: 1.6807 || Test loss: 1.7689 || Train accuracy: 59.52% || Test accuracy: 53.8%
Epoch: 90 || Train loss: 1.58648 || Test loss: 1.68288 || Train accuracy: 63.7% || Test accuracy: 57.6%

Sequential mnist training time: 79.0261 seconds

Truth: 9 || Pred: 4






            .#
          .###
          ####.
         .######.
         ########
        .##  .####
        .#.   ####
        .#.   ###.
        .## ..####
         ###### ##.
          ###.  .##
           ..    .#.
                 .##
                  .#.
                   ##
                   .#.
                    ##
                     #.
                     ##
                      #.



OpenMP mnist dataset test

Epoch: 0 || Train loss: 2.33547 || Test loss: 2.33067 || Train accuracy: 10.64% || Test accuracy: 12.2%
Epoch: 50 || Train loss: 2.06295 || Test loss: 2.10641 || Train accuracy: 37.3% || Test accuracy: 29.7%
Epoch: 100 || Train loss: 1.73838 || Test loss: 1.82255 || Train accuracy: 59.34% || Test accuracy: 53.7%
Epoch: 150 || Train loss: 1.35909 || Test loss: 1.47436 || Train accuracy: 69.28% || Test accuracy: 62.4%
Epoch: 200 || Train loss: 1.03019 || Test loss: 1.15745 || Train accuracy: 76.74% || Test accuracy: 69.8%
Epoch: 250 || Train loss: 0.804067 || Test loss: 0.931436 || Train accuracy: 80.98% || Test accuracy: 74.9%

OpenMP mnist training time: 27.6625 seconds

Truth: 9 || Pred: 9






            .#
          .###
          ####.
         .######.
         ########
        .##  .####
        .#.   ####
        .#.   ###.
        .## ..####
         ###### ##.
          ###.  .##
           ..    .#.
                 .##
                  .#.
                   ##
                   .#.
                    ##
                     #.
                     ##
                      #.



CUDA mnist dataset test

Epoch: 0 || Train loss: 2.49682 || Test loss: 2.41311 || Train accuracy: 13.24% || Test accuracy: 14.6%
Epoch: 100 || Train loss: 1.49957 || Test loss: 1.53557 || Train accuracy: 64.78% || Test accuracy: 63.3%
Epoch: 200 || Train loss: 0.805449 || Test loss: 0.897347 || Train accuracy: 80.5% || Test accuracy: 75.8%
Epoch: 300 || Train loss: 0.56172 || Test loss: 0.663556 || Train accuracy: 85.48% || Test accuracy: 80.8%
Epoch: 400 || Train loss: 0.456369 || Test loss: 0.559097 || Train accuracy: 87.46% || Test accuracy: 83.6%
Epoch: 500 || Train loss: 0.396975 || Test loss: 0.501145 || Train accuracy: 88.66% || Test accuracy: 85.2%
Epoch: 600 || Train loss: 0.357763 || Test loss: 0.464023 || Train accuracy: 89.7% || Test accuracy: 86.1%

CUDA mnist training time: 7.91312 seconds

Truth: 9 || Pred: 9






            .#
          .###
          ####.
         .######.
         ########
        .##  .####
        .#.   ####
        .#.   ###.
        .## ..####
         ###### ##.
          ###.  .##
           ..    .#.
                 .##
                  .#.
                   ##
                   .#.
                    ##
                     #.
                     ##
                      #.


C:\Users\ASUS\source\repos\NeuralNetwork\x64\Release\NeuralNetwork.exe (process 14228) exited with code 0 (0x0).
To automatically close the console when debugging stops, enable Tools->Options->Debugging->Automatically close the console when debugging stops.
Press any key to close this window . . .
