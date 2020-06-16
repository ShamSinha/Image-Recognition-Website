# Image Recognition Website


## In Windows 
1. [Install Anaconda ](https://docs.anaconda.com/anaconda/install/)
        
2. Now we install all the packages required to run our website using Anaconda prompt command line
   - Install OpenCV library
   
         conda install -c conda-forge opencv
   - Install Tensorflow-cpu  
   
         conda install tensorflow-cpu
   - Open SPYDER 
         
         spyder
 
         
3. Now clone this github master branch repo in your local machine.
4. Run app.py in spyder 


## Landmarks Recognition Model


**These are the 30 landmarks that our Landmark recognition model written in Keras can recognise with 77.55% accuracy in validation set which comprise of 590 images and training set comprise of 5300 images.**

Procedures to load our **Landmark recognition Model** in Google Colab 
1. Open this link(https://drive.google.com/file/d/1VtiL6fqkCT_ehzRP2lBPo71hqL-wyC5c/view?usp=sharing) to get our Landmark recognition model in google drive.
2. Open Shared with me within Google Drive and use Add Shortcut to Drive option to make shortcut of ***Model30v2_77.h5*** file in My Drive.
3. Open Google Colab and run

        from google.colab import drive
        drive.mount('/content/drive')
        
4. After successful mount the drive run this to load model
        
        from keras.models import load_model
        model = load_model('/content/drive/My Drive/Model30v2_77.h5')
        

Procedures to load Landmark dataset in Google Colab used for Training and Testing of our **Landmark recognition Model**

1. Open this [link](https://drive.google.com/file/d/1VtiL6fqkCT_ehzRP2lBPo71hqL-wyC5c/view?usp=sharing).
2. Open Shared with me within Google Drive and use Add Shortcut to Drive option to make shortcut of ***Landmark_data30.npz*** dataset file in My Drive. 

*Dataset files is after get rescaled by 1.0/255 so therefore do not rescale again if you use this dataset for training*.

4. Open google Colab and run
        
        from google.colab import drive
        drive.mount('/content/drive')
        
5. After successful mount the drive run this to load dataset
        
        npzfile = np.load('/content/drive/My Drive/Landmark_data30.npz')
        X = npzfile['arr_0']
        Y = npzfile['arr_1']
        np.random.seed(150)
        np.random.shuffle(X)
        np.random.seed(150)
        np.random.shuffle(Y)
        
5. You can use this code in Colab to split the dataset into train, dev and test set 

        import math
        m = X.shape[0]
        print("X.shape[0]: " + str(m))
        dev_set_percentage = int(input('Enter percentage_dev_set :'))
        test_set_percentage = int(input('Enter percentage_test_set :'))
        size_dev_set = math.floor(m*(dev_set_percentage/100))
        size_test_set = math.floor(m*(test_set_percentage/100))
        size_train_set = m - size_dev_set - size_test_set

        X_train = X[0:size_train_set]
        Y_train = Y[0:size_train_set]

        X_dev = X[size_train_set:size_train_set+size_dev_set]
        Y_dev = Y[size_train_set:size_train_set+size_dev_set]

        X_test = X[size_train_set+size_dev_set-1:m-1]
        Y_test = Y[size_train_set+size_dev_set-1:m-1]

        print ("number of training examples = " + str(X_train.shape[0]))
        print ("number of dev examples = " + str(X_dev.shape[0]))
        print ("number of test examples = " + str(X_test.shape[0]))
        print ("X_train shape: " + str(X_train.shape))
        print ("Y_train shape: " + str(Y_train.shape))
        print ("X_dev shape: " + str(X_dev.shape))
        print ("Y_dev shape: " + str(Y_dev.shape))
        print ("X_test shape: " + str(X_test.shape))
        print ("Y_test shape: " + str(Y_test.shape))
 
        
        
        
### List of Landmarks our model can recognize
1. Angkor Wat
2. Arc de Triomphe
3. Big Ben
4. Burj Al Arab
5. Burj Khalifa
6. Charminar
7. Chichen Itza
8. Christ The Redeemer
9. Colosseum
10. Effiel tower
11. Empire state building
12. Golden Gate Bridge
13. Golden Temple
14. Hawa Mahal
15. India Gate
16. Leaning Tower of Pisa
17. Lotus temple
18. Mount Rushmore
19. Petronas Tower
20. Pyramid of Giza
21. Qutub Minar
22. Red Fort
23. Statue of Liberty
24. Stone Henge
25. Sydney opera house
26. Taj Hotel
27. Taj Mahal
28. Tower Bridge
29. Victoria Memorial
30. White house



![](https://github.com/ShamSinha/Image-Recognition-Website/blob/branch1/images/1.png?raw=true)
![](https://github.com/ShamSinha/Image-Recognition-Website/blob/branch1/images/2.png?raw=true)


