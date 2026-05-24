# NeuralNetwork

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

Epoch: 0 || Train loss: 0.675588 || Test loss: 0.799319 || Train accuracy: 64.1148% || Test accuracy: 34% || Time: 1.34004 sec
Epoch: 100 || Train loss: 0.435139 || Test loss: 0.532536 || Train accuracy: 73.6842% || Test accuracy: 78% || Time: 117.136 sec
Epoch: 200 || Train loss: 0.323595 || Test loss: 0.504824 || Train accuracy: 84.2105% || Test accuracy: 80% || Time: 235.108 sec
Epoch: 300 || Train loss: 0.237676 || Test loss: 0.506617 || Train accuracy: 88.9952% || Test accuracy: 80% || Time: 354.598 sec
Epoch: 400 || Train loss: 0.170707 || Test loss: 0.526326 || Train accuracy: 94.2584% || Test accuracy: 80% || Time: 469.336 sec
Epoch: 500 || Train loss: 0.117655 || Test loss: 0.560655 || Train accuracy: 98.5646% || Test accuracy: 78% || Time: 582.198 sec
Epoch: 600 || Train loss: 0.0777854 || Test loss: 0.608299 || Train accuracy: 99.5215% || Test accuracy: 78% || Time: 694.25 sec
Epoch: 700 || Train loss: 0.053182 || Test loss: 0.669524 || Train accuracy: 100% || Test accuracy: 74% || Time: 805.184 sec

Sequential cat training time: 805.236 seconds

Truth: 1 || Pred: 0

OpenMP cat dataset test

Epoch: 0 || Train loss: 0.758618 || Test loss: 1.43023 || Train accuracy: 38.756% || Test accuracy: 34% || Time: 0.0594862 sec
Epoch: 100 || Train loss: 0.45634 || Test loss: 0.831209 || Train accuracy: 87.0813% || Test accuracy: 38% || Time: 6.83982 sec
Epoch: 200 || Train loss: 0.353437 || Test loss: 0.805209 || Train accuracy: 89.4737% || Test accuracy: 50% || Time: 13.7209 sec
Epoch: 300 || Train loss: 0.270849 || Test loss: 0.781404 || Train accuracy: 93.3014% || Test accuracy: 60% || Time: 20.622 sec
Epoch: 400 || Train loss: 0.200205 || Test loss: 0.762845 || Train accuracy: 96.6507% || Test accuracy: 62% || Time: 27.5659 sec
Epoch: 500 || Train loss: 0.136887 || Test loss: 0.738544 || Train accuracy: 98.5646% || Test accuracy: 64% || Time: 34.5688 sec
Epoch: 600 || Train loss: 0.0884581 || Test loss: 0.710055 || Train accuracy: 98.5646% || Test accuracy: 68% || Time: 41.566 sec
Epoch: 700 || Train loss: 0.0591217 || Test loss: 0.686301 || Train accuracy: 99.5215% || Test accuracy: 72% || Time: 48.8558 sec

OpenMP cat training time: 48.9013 seconds

Truth: 1 || Pred: 0

CUDA cat dataset test

Epoch: 0 || Train loss: 0.73747 || Test loss: 0.734932 || Train accuracy: 65.5502% || Test accuracy: 42% || Time: 0.00626 sec
Epoch: 100 || Train loss: 0.492689 || Test loss: 0.533569 || Train accuracy: 70.8134% || Test accuracy: 76% || Time: 0.657742 sec
Epoch: 200 || Train loss: 0.397199 || Test loss: 0.479589 || Train accuracy: 77.0335% || Test accuracy: 80% || Time: 1.2295 sec
Epoch: 300 || Train loss: 0.317063 || Test loss: 0.457964 || Train accuracy: 83.2536% || Test accuracy: 82% || Time: 1.74728 sec
Epoch: 400 || Train loss: 0.245859 || Test loss: 0.455132 || Train accuracy: 90.4306% || Test accuracy: 82% || Time: 2.25515 sec
Epoch: 500 || Train loss: 0.185374 || Test loss: 0.46554 || Train accuracy: 94.2584% || Test accuracy: 82% || Time: 2.75001 sec
Epoch: 600 || Train loss: 0.133849 || Test loss: 0.487701 || Train accuracy: 97.1292% || Test accuracy: 82% || Time: 3.22844 sec
Epoch: 700 || Train loss: 0.0900744 || Test loss: 0.526929 || Train accuracy: 99.5215% || Test accuracy: 82% || Time: 3.70941 sec

CUDA cat training time: 3.91586 seconds

Truth: 1 || Pred: 0

Sequential mnist dataset test

Epoch: 0 || Train loss: 2.37501 || Test loss: 2.38058 || Train accuracy: 8.32% || Test accuracy: 8.5% || Time: 0.856481 sec
Epoch: 100 || Train loss: 1.50541 || Test loss: 1.58992 || Train accuracy: 65.52% || Test accuracy: 60.8% || Time: 78.1534 sec
Epoch: 200 || Train loss: 0.832999 || Test loss: 0.928163 || Train accuracy: 80.48% || Test accuracy: 77.4% || Time: 152.709 sec
Epoch: 300 || Train loss: 0.570633 || Test loss: 0.670409 || Train accuracy: 85.88% || Test accuracy: 82.7% || Time: 228.38 sec
Epoch: 400 || Train loss: 0.454798 || Test loss: 0.559935 || Train accuracy: 88.26% || Test accuracy: 84.2% || Time: 303.088 sec
Epoch: 500 || Train loss: 0.39094 || Test loss: 0.501023 || Train accuracy: 89.36% || Test accuracy: 85.4% || Time: 377.613 sec
Epoch: 600 || Train loss: 0.349278 || Test loss: 0.463714 || Train accuracy: 90.3% || Test accuracy: 86.3% || Time: 455.109 sec
Epoch: 700 || Train loss: 0.31902 || Test loss: 0.437212 || Train accuracy: 91.3% || Test accuracy: 86.2% || Time: 527.193 sec

Sequential mnist training time: 527.254 seconds

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
                  ##
                   #.
                   ##
                    #.



OpenMP mnist dataset test

Epoch: 0 || Train loss: 2.38729 || Test loss: 2.35986 || Train accuracy: 10.8% || Test accuracy: 11.4% || Time: 0.0668349 sec
Epoch: 100 || Train loss: 1.42917 || Test loss: 1.50612 || Train accuracy: 68.08% || Test accuracy: 62.8% || Time: 8.19211 sec
Epoch: 200 || Train loss: 0.802116 || Test loss: 0.909274 || Train accuracy: 82.08% || Test accuracy: 77.5% || Time: 16.4572 sec
Epoch: 300 || Train loss: 0.558314 || Test loss: 0.670035 || Train accuracy: 86.24% || Test accuracy: 81.8% || Time: 24.7624 sec
Epoch: 400 || Train loss: 0.448335 || Test loss: 0.559289 || Train accuracy: 88.38% || Test accuracy: 84.4% || Time: 33.2435 sec
Epoch: 500 || Train loss: 0.386202 || Test loss: 0.497086 || Train accuracy: 89.92% || Test accuracy: 85.8% || Time: 41.6686 sec
Epoch: 600 || Train loss: 0.345795 || Test loss: 0.457831 || Train accuracy: 90.82% || Test accuracy: 86.9% || Time: 50.1316 sec
Epoch: 700 || Train loss: 0.31623 || Test loss: 0.43083 || Train accuracy: 91.38% || Test accuracy: 87.6% || Time: 58.7461 sec

OpenMP mnist training time: 58.759 seconds

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
                  ##
                   #.
                   ##
                    #.



CUDA mnist dataset test

Epoch: 0 || Train loss: 2.93916 || Test loss: 2.54614 || Train accuracy: 9.96% || Test accuracy: 12.1% || Time: 0.0771189 sec
Epoch: 100 || Train loss: 1.08881 || Test loss: 1.16974 || Train accuracy: 76.02% || Test accuracy: 71.3% || Time: 1.37293 sec
Epoch: 200 || Train loss: 0.651034 || Test loss: 0.757759 || Train accuracy: 83.92% || Test accuracy: 79.6% || Time: 2.46619 sec
Epoch: 300 || Train loss: 0.494483 || Test loss: 0.607002 || Train accuracy: 87.36% || Test accuracy: 82.5% || Time: 3.53228 sec
Epoch: 400 || Train loss: 0.417483 || Test loss: 0.531262 || Train accuracy: 88.92% || Test accuracy: 85% || Time: 4.59855 sec
Epoch: 500 || Train loss: 0.370601 || Test loss: 0.485533 || Train accuracy: 89.94% || Test accuracy: 85.6% || Time: 5.66725 sec
Epoch: 600 || Train loss: 0.33824 || Test loss: 0.455087 || Train accuracy: 90.7% || Test accuracy: 86.8% || Time: 6.73093 sec
Epoch: 700 || Train loss: 0.313933 || Test loss: 0.433237 || Train accuracy: 91.38% || Test accuracy: 87.5% || Time: 7.8377 sec

CUDA mnist training time: 7.85793 seconds

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
                  ##
                   #.
                   ##
                    #.


C:\Users\ASUS\source\repos\NeuralNetwork\x64\Release\NeuralNetwork.exe (process 14772) exited with code 0 (0x0).
To automatically close the console when debugging stops, enable Tools->Options->Debugging->Automatically close the console when debugging stops.
Press any key to close this window . . .
