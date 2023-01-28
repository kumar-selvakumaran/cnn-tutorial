```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

    /kaggle/input/digit-recognizer/sample_submission.csv
    /kaggle/input/digit-recognizer/test.csv
    /kaggle/input/digit-recognizer/train.csv
    


<h1><span style ="font-family:charcoal;font-size:32px;color:Blue;;"><b>   Visual Guide to Convolution Neural Networks (CNNs)&nbsp; <span style="color:Tomato;" >FOR EVERYONE&nbsp;</span></b></span></h1>

<img src="https://thumbs.gfycat.com/HelpfulScratchyArcticseal.webp">

<h1 style="color:Tomato;"><b><i>INTRODUCTION</i></b></h1>  

<h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:18px;color:#334761">"How do we actually identify the objects we see?" this is one of the basic questions we have to ask ourselves before we try to implement it as CNNs. to answer this question let us look at an example. </h4>


<img src="https://upload.wikimedia.org/wikipedia/commons/0/09/TheCheethcat.jpg" width="300" height="400">

<h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:18px;color:#334761"><ul><li>Above is the picture of a Cheetah , clearly , but how do we identify it as a cheetah ? it is done by noticing the various&nbsp;<i>features</i>&nbsp; that the cheetah posseses , such as it's colour , the spots on its coat , the 'tear lines' that run in between its face etc.</li><li>Thus we understand that , objects are classified/identified by us based on its distinctive characteristics /<i>&nbsp;features</i>&nbsp;.</li><li>the result of implementing this concept of identifiying objects with the help of it's characteristics is what lead to the development of <span style ="color:Tomato"><i>&nbsp;&nbsp;Convolution Neural Networks (CNNs)</i></span></li><li>Just like how our eyes scan through an image to look for features , CNNs scan through through the image to find various features. </li><li>Although the method is same , the features that are identified by CNNs maybe different than what humans use to classify an object . this is what causes mis-classification , but in some cases this behaviour also allows models to identify objects from images that even humans cannot easily interpret!</li><li>if u think about it , this behaviour can also be found among different people , it is not that all of us might interpret an object the same way, everyone thinks differently and hence we might have a difference in opinion , given below are some examples that will help u understand what I'm talking about</li></ul> </h4>

<img src="https://i.pinimg.com/236x/68/72/83/6872834a4d21c9ea4e332deb16faa9db.jpg" width="300" height="400">

<h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:18px;color:#334761"><p>what do u see in the above image? ;)</p>
<p>This is the famous old woman / young lady illusion . The  image can be interpreted in to ways :
<ol><li>the young lady : there is a young lady with a sharp jawline and curly brown short hair looking into the screen , u can see her small nose , next to her eyleash ,u can see her left ear ,  and u see that she is wearing a pinkish necklace neckless . </li><li>The Old Lady : what appeared to be the young lady's jawline is now the nostrill outline of the old woman's nose, who is facing downwards ,  with her left eye closed (the left eye appears as the young lady's left ear ). U can see her thin lips ( which appeared as the young lady's necklace ) . she has an overall 'witchy' appearance . the little wart on her nose is what appeared as the young lady's nose above which u can see an eyelash.</li></ol></p><p>It might take a while to notice both the old woman and the young lady . from this example , we can see how focussing on different features leads to different classification of the same image . Now that we have seen the importance of features to classify the image. let us move on to <i>how it is done is CNNs</i></p> 
    

<p><h1 style="color:Tomato;"><b><i>WHAT IS AN IMAGE ? (how does a computer comprehend it?)</i></b></h1></p>
<h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:18px;color:#334761"><ul><li>Images are basically , either a 2d array (row x column) of values from 0-255 for a gray-scale image or a 3d  array (row x column x channel)  of values from 0-255 for a colour image.<br><br> </li><li>the 3 dimentions are : <br><br> <ol><li> rows of pixels </li><li>columns of pixels</li><li>(if coloured) how red/green/blue an image is respectively (all colours are intricate combinations of brightnesses of the  three basic colours red green and blue).</li></ol><br><li>the values (0-255) corresponds to the brightness of that particular pixel . </li></ul><br>
<img src="https://static.packt-cdn.com/products/9781789613964/graphics/e91171a3-f7ea-411e-a3e1-6d3892b8e1e5.png" width="500" height="600"></h4>



<h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:18px;color:#334761"> let us now move on to understanding the elements of CNNs</h4>

<p><h1 style="color:Tomato;"><b><i>ELEMENTS OF CNNs : </i></b></h1></p>
<P>
<h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><ol><li><span style="font-size:30px;color:#334761;color:Green">&nbsp;&nbsp;<b><i>FEATURE DETECTORS</i></b></span> &nbsp;&nbsp;(aka filters/kernels/array of weights) &nbsp; : <br><br>
    <img src="https://media1.giphy.com/media/2kXLNQypdX9O1A3zxX/giphy.gif" width="400" height="300">
    <br><br><ul><li><span style="font-size:20px;color:#334761;color:Tomato">&nbsp;&nbsp;<b><i>FEATURE DETECTORS </i></b></span>&nbsp;&nbsp; are small arrays ( usually of dimentions 3x3) that contain values that act as  multipliers . the pixel values of the image are changed when filters are applied to it . This new image with changed pixel values will reveal different features based on the filter that is applied .<br><br></li><li>how is a filter applied ? : <br><br><ul><li>the filter is slided accross an image in a step by step fashion . each step is called a stride . At each stride ,the values of the sub-array of the image corresponding to the position of the filter is  element-wise multiplied to the values of the filter , and these new values are summed up to give one single final value . </li><li> the resultant value of each stride/step is put into a new array called the <b><i>feature map . The process of how this feature map is formed is called <span style="font-size:20px;color:#334761;color:Tomato">&nbsp;&nbsp;<b><i>CONVOLUTION</i></b></span></i></b></li><li>the resultant matrix (feature map) is a compressed form of the image reducing the rows and column dimention by a value of 2 , that is if a 3x3 filter with stride = 1(row/columns increasing by 1 step) is moved along a 15x15 image , the feature map will be of size 13x13 . if we want a feature map of the same dimention as the input image, we can add the required amount of padding (columns / rows of 0s). </ul><br><br><br>
    <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><span style="font-size:25px;color:#334761;color:Green">&nbsp;&nbsp;<b><i>2D CONVOLUTION VISUAL (step by step) (without padding)</i></b></span>
    <br><br><br>
    <img src="https://aigeekprogrammer.com/wp-content/uploads/2019/12/CNN-filter-animation-1.gif" width="900" height="800">
    <br><br><br>
    </h4>
    <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><span style="font-size:25px;color:#334761;color:Green">&nbsp;&nbsp;<b><i>3D CONVOLUTION VISUAL (step by step)</i></b></span>
    <br><br><br>
    <img src="https://www.researchgate.net/profile/Santhiya_Rajan/post/How_will_channels_RGB_effect_convolutional_neural_network/attachment/5c67b72d3843b0544e664e12/AS%3A726829115666434%401550300973344/download/cnn_1.gif" width="900" height="800">
    <br><br><br>
    </h4>
    <li>One part of CNNs are groups of such filters , these groups of filters are called<span style="font-size:20px;color:#334761;color:Tomato">&nbsp;&nbsp;<b><i>CONVOLUTION LAYERS</i></b></span></li>
    <li>below is a visual representation of how feature maps of each [filter / array of weights] is stacked .
        <br><br><br>
    <img src="https://i.stack.imgur.com/FjvuN.gif" width="800" height="700">
    <br><br><br>
   </li><li>Now that we have understood what feature detectors are and what the Convulution layer is  , we will move on to the next element of CNNs</li></ul><br><br><br>           <li> <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><span style="font-size:30px;color:#334761;color:Green">&nbsp;&nbsp;<b><i>RELU function :</i></b></span>&nbsp;&nbsp;<br><br><br>
    <img src="https://machinelearningknowledge.ai/wp-content/uploads/2019/08/Relu-Activation-Function.gif" width="600" height="500">
    <br><br><ul><li>
    <span style="font-size:20px;color:#334761;color:Tomato">&nbsp;&nbsp;<b><i>RELU FUNCTION</i></b></span>&nbsp;&nbsp;(REctifier Linear Unit) : it is the most common activation function used in Neural Networks .</li><li>WHY RELU ? : <ul><li>images are genrally non linear by nature , this non linearity should be maintained to keep the performance of CNN optimal .</li><li> when convolution is done , there is a risk of linearity in the feature map , this is where RELU comes in ,  using the RELU as the activation function ensures that this non linearity is maintatined . </li> </ul></li></ul><br><br>
    <li><h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><span style="font-size:30px;color:#334761;color:Green">&nbsp;&nbsp;<b><i>POOLING :</i></b></span>&nbsp;&nbsp;<br><br><br>
    <img src="https://media1.tenor.com/images/f181464c1be3f16db829c46966eab6fd/tenor.gif?itemid=3452012" width="400" height="300">
    <br><br>
        <ul><li>WHAT IS POOLING ? AND HOW IS IT USEFULL ? : <br><br><ul><li>Pooling is a technique that is used to controll overfitting , by providing <span style="font-size:20px;color:#334761;color:Tomato">&nbsp;&nbsp;<b><i>SPATIAL INVARIANCE</i></b></span>&nbsp;&nbsp;and compressing the data further . ( keeps the values that indicate the positions of the features and removes rest of the values )</li><li><span style="font-size:20px;color:#334761;color:Tomato">&nbsp;&nbsp;<b><i>SPATIAL INVARIANCE : </i></b></span>&nbsp;&nbsp; it is a property that allows the CNN to detect features even if they are slightly displaced or disoriented .</li><li>By compressing the data without loosing much information , it reduces the computional load on CNNs.</li></ul>
            <br><br><li>HOW IS POOLING DONE ? : <br><br><ul><li>Like Convolution , pooling is also done by striding along the feature maps. </li><li>In each stride , a single value is calculated that will replace the values of that particular patch in that stride</li><li>The value calculated is based on the type of pooling that is done .</li><li> the resultant values are packed into an array just like when feature maps were made . this process is continuoes for 'n' feature maps resulting in 'n' pooled feature maps .</li><li>This layer that does the pooling function is called the <i> pooling layer </i> </li><li> The most common and preffered type of pooling is <span style="font-size:20px;color:#334761;color:Tomato">&nbsp;&nbsp;<b><i>MAX POOLING</i></b></span>&nbsp;&nbsp;(aka down-sampling). <br><span style="font-size:20px;color:#334761;color:Tomato"><b><i>MAX POOLING</i></b></span> &nbsp;&nbsp;:  for each stride the maximum of the values belonging to that particular patch is selected . hence the pooled feature map will contain the maximum values of all the the patches . Larger values indicate the presence of features , hence by keeping the maximum values , we compress the feature map without loosing inportant information . </li><li>other types of pooling techniques are : <br><br><ul><li><i>AVERAGE POOLING</i>&nbsp;&nbsp;(aka sub-sampling) : pooling the feature map by calculating the average of values of each patch of the featre map .</li><li><i>GLOBAL POOLING</i>&nbsp;&nbsp;: special case of down-sampling where [stride/patch] size = size of feature map , That is , the feature map is reduced to a single value .</li>
    <li><i>SUM POOLING</i>&nbsp;&nbsp;:the sum of all the values of each patch of the feature map is calculated to make the pooled feature map. <br><br><br>
    <img src="https://www.bouvet.no/bouvet-deler/understanding-convolutional-neural-networks-part-1/_/attachment/inline/e60e56a6-8bcd-4b61-880d-7c621e2cb1d5:6595a68471ed37621734130ca2cb7997a1502a2b/Pooling.gif" width="900" height="800">
    <br><br>
            </li></ul></li></ul></li></ul>
        <ul><li>
            Till now we have come across convolution layers , the relu function and pooling .</li><li>During application , the relu function is built into the convolution layers . </li><li>we add multiple convolutional and pooling layers , to extract more advanced features and reduce the data to a very efficiently compressed format . one step remains in the convolutional neural network and that is the ... <br><br>  
            </li></ul> <li> <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><span style="font-size:30px;color:#334761;color:Green">&nbsp;&nbsp;<b><i>FLATTENING LAYER :</i></b></span>&nbsp;&nbsp;<br><br><br>
    <img src="https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/b4b71c37-c0e7-45b1-9b28-dd1e52c99fb9/d8hlsgu-9ab3e032-ecfa-4832-9498-d20c54ba9a19.gif?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOiIsImlzcyI6InVybjphcHA6Iiwib2JqIjpbW3sicGF0aCI6IlwvZlwvYjRiNzFjMzctYzBlNy00NWIxLTliMjgtZGQxZTUyYzk5ZmI5XC9kOGhsc2d1LTlhYjNlMDMyLWVjZmEtNDgzMi05NDk4LWQyMGM1NGJhOWExOS5naWYifV1dLCJhdWQiOlsidXJuOnNlcnZpY2U6ZmlsZS5kb3dubG9hZCJdfQ.hI41Awsbr9E7m1QKJhpRrCf3Cn65695bTFwbMh9zPFQ" width="400" height="300">
    <br><br> <span style="font-size:20px;color:#334761;color:Tomato"><b><i>FLATTENING LAYER</i></b></span>&nbsp;&nbsp;:<br><br> <ul><li>Now that we have the data in a very compressed format , we want to convert it in such a way that it can be passed as input to a machine learning model (prefferably a Artificial Neural Network ) . </li><li>The flattening layer completes this task by reshaping and concatenating each of the final feature maps into a looong vector as shown in the image below.
        <br><br>
    <img src="https://www.researchgate.net/profile/Budiman_Minasny/publication/334783857/figure/fig4/AS:786596169269249@1564550549811/Illustration-of-flatten-layer-that-is-connecting-the-pooling-layers-to-the-fully.png" width="600" height="500">
        <br><br></li></ul>

 <span style="font-size:20px;color:#334761;color:Tomato"><b><i>NOTE  </i></b></span>&nbsp;&nbsp;: just like we preprocess data in any machine learning problem , we need to preprocess our data for CNNs , this is done by normalizing the pixel values of the image so that it lies from 0 to 1. this helps the model's complex calculations and in reducing computational load , hence decreases the training time .  <br><br>
        
 And finally ...  
<br><br><li><span style="font-size:20px;color:#334761;color:Tomato"><b><i>PREDICTION MODELLING</i></b></span>&nbsp;&nbsp;: artifcial neural networks are the most common models used to make predictions taking the flattened vector as input . </li></ol>

<br> <span style="font-size:20px;color:#334761;color:Tomato"><b><i>Some important terms </i></b></span>&nbsp;&nbsp;:
<ul><li><span style="font-size:18px;color:#334761;color:Green"><b><i>SOFTMAX FUNCTION</i></b></span>&nbsp;&nbsp;: it is the activation function of the final predictive layer of a multi-classification CNN model . it predicts the probabiliy of the image belonging to each of the classes . the class with the highest probability is considered as the prediction.</li>
    <li><span style="font-size:18px;color:#334761;color:Green"><b><i>CATEGORICAL CROSS-ENTROPY</i></b></span>&nbsp;&nbsp;: It is one of the most preffered loss function used for single label classification in CNNs .
        <br><br><br></li></ul></li>
We will bring all the different elements together and analyse the complete framework of a CNN now . 
    <img src="https://static.wixstatic.com/media/d77a32_076f341258d24f47852625faaa726620~mv2.jpeg/v1/fill/w_1422,h_760,al_c,q_85,usm_0.66_1.00_0.01/1_uAeANQIOQPqWZnnuH-VEyw.webp" width="1200" height="1100">
        <br><br>
<span style="font-size:18px;color:#334761;color:Green"><b><i>FRAMEWORK DETAILS </i></b></span>&nbsp;&nbsp;: The above framework contains 2 convolution layer [conv_1 , conv_2] , each of these layers followed by a pooling layer . it contains an artificial NN for predictive modelling that takes the final flattened vector as input .
    <br><br><li><span style="font-size:18px;color:#334761;color:Green"><b><i>input layer</i></b></span>&nbsp;&nbsp;: a gray-scale image of dimentions [28x28x1]<br><br></li><li><span style="font-size:18px;color:#334761;color:Green"><b><i>conv_1</i></b></span>&nbsp;&nbsp;: contains 'n1' filters , each of dimention 5x5 , stride_steps (row/column movement per stride) = 1 . Converts the input layer in n1 [24x24] sized feature maps since it has n1 filters.</li><br><li><span style="font-size:18px;color:#334761;color:Green"><b><i>conv_2</i></b></span>&nbsp;&nbsp;: contains ['n2'/'n1'] filters , each of dimention 5x5 , stride_steps = 1 .Converts the output of pooled-conv_1 into n2 [8x8] sized feature maps since it has n2/n1 filters ( each of n1 filters are convoluted into [n2/n1] feature maps .</li><br><li><span style="font-size:18px;color:#334761;color:Green"><b><i>max pooling layer</i></b></span>&nbsp;&nbsp;:patch size = 2x2 , stride_steps = 2 .reduces size of the feature maps by 75%.</li><br><li><span style="font-size:18px;color:#334761;color:Green"><b><i>flattening layer</i></b></span>&nbsp;&nbsp;: converts the final 'n2' feature maps of dimention 4x4 , into a single single vector of length (4 x 4 x n2).</li>
<br><li><span style="font-size:18px;color:#334761;color:Green"><b><i>predicive modelling : </i></b></span>&nbsp;&nbsp; <ul style="list-style-type:circle; margin-left: 10px;" ><br><li>input layer : flattened vector </li><li> hidden layer of n3 nodes </li><li> output layer of 10 node (for classes [0 to 10]) </ul>
    <br> 
   
<span style="font-size:30px;color:#334761;color:Green"><b><i>As of now we have succesfully learnt the basics of CNNs . </i></b></span>&nbsp;&nbsp;<span style='font-size:100px;'>&#129346;</span></h4>

 




<h1  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:Blue"> It is finally time for us to move on the application part of CNNs 

<h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761">
<span style="font-size:38px;color:#663399;font-family:Monaco,sans-serif;">MNIST classifcation </span>&nbsp;&nbsp;
    <br><br><p>We will use the architecture shown in the framework above for easy reference and understanding.</p>



<h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><span style="font-size:38px;color:orange;9;font-family:Monaco,sans-serif;">UNDERSTANDING THE DATA </span>&nbsp;&nbsp;:<br><br><ul><li>The training data set, (train.csv), has 785 columns. </li><li>The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.</li><li>
Each pixel column in the training set has a name like pixel 'x', where 'x' is an integer between 0 and 783, inclusive. </li><li>To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero).</li><li>The test data set, (test.csv), is the same as the training set, except that it does not contain the "label" column.</li><li>
Your submission file should be in the following format: For each of the 28000 images in the test set, output a single line containing the ImageId and the digit you predict. </li><br><b>format :<br><br>ImageId,Label<br></b>
1,3<br>
2,7<br>
3,8 <br>    
    (27997 more lines)</li></ul><br><br>
 <span style="font-size:28px;color:Green;font-family:Monaco,sans-serif;">let us begin by importing the necassary libraries , we will use keras implementation of CNNs in this notebook .</span>&nbsp;&nbsp;
 



```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D,MaxPooling2D
from keras.models import Model
from keras.layers import Input,InputLayer
from keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import h5py
```

    Using TensorFlow backend.
    

<h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><ol><li><span style="font-size:38px;color:orange;9;font-family:Monaco,sans-serif;">DATA PREPROCESSING </span>&nbsp;&nbsp;:
    <br><br><ol><li><span style="font-size:24px;color:green;9;font-family:Monaco,sans-serif;">ISOLATING THE LABELS FOR EACH IMAGE </span>&nbsp;&nbsp; :<br>since the first column contains the target (label of the image) . we will isolate it from the training data. we will also one-hot-encode the label beacuse this is the desired format of labels for keras when softmax function is used from predictions .</li><br><li><span style="font-size:24px;color:green;9;font-family:Monaco,sans-serif;">RESHAPING THE INPUT INTO A KERAS ACCEPTABLE FORMAT </span>&nbsp;&nbsp; :<br> we will transform the input into an array of dimention [length of image x breadth of image x no.of images] which is acceptable by the keras implementation.<br><br><li><span style="font-size:24px;color:green;9;font-family:Monaco,sans-serif;">SCALING</span>&nbsp;&nbsp; :<br>we will divide all the pixel values by 255 so that all the pixel values will lie from [0-1] .</li></ol>
<br>We will now implement data preprocessing using the function data_procc()

   
    



```python
def prep_data(raw):
    img_rows, img_cols = 28, 28
    num_classes = 10
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    x = raw[:,1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y , img_rows, img_cols, num_classes, num_images

mnist_file = "/kaggle/input/digit-recognizer/train.csv"
mnist_data = np.loadtxt(mnist_file, skiprows=1, delimiter=',')
x, y, img_rows, img_cols, num_classes, num_images= prep_data(mnist_data)
print('input shape : {}'.format(x.shape))
print('output shape : {}'.format(y.shape))
```

    input shape : (42000, 28, 28, 1)
    output shape : (42000, 10)
    

<h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><span style="font-size:28px;color:#334761;9;font-family:Monaco,sans-serif;">now that we have completed data preprocessing we will move on to buliding out model .</span>&nbsp;&nbsp;
    <br><br><br>
    since we decided our model will be the same as the framework we have seen before , let us look at it again .
    <br><br><br>
        <img src="https://static.wixstatic.com/media/d77a32_076f341258d24f47852625faaa726620~mv2.jpeg/v1/fill/w_1422,h_760,al_c,q_85,usm_0.66_1.00_0.01/1_uAeANQIOQPqWZnnuH-VEyw.webp" width="1200" height="1100">

<h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><span style="font-size:38px;color:orange;9;font-family:Monaco,sans-serif;">ARCHITECTURE</span>&nbsp;&nbsp;: <br><br><br><span style="font-size:28px;color:green;9;font-family:Monaco,sans-serif;"><br><br><br>1st convolutional layer</span>&nbsp;&nbsp;: <br><br> <ul><li>no of filters (n1) = 16<li>filter dimentions = [5x5]</li><li>we will use the relu function as the activation function for all the layers except the output layer.</li><li>we need to pass the dimentions of the input as input shape , only for the first layer</li>   


```python
#initializing our model
cnn_pilot = Sequential()

#adding the first convolutional layer
cnn_pilot.add(Conv2D(filters = 16,
                     kernel_size = 5,
                     activation = 'relu',
                     input_shape = (img_rows, img_cols, 1)))
```

<h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><span style="font-size:28px;color:green;9;font-family:Monaco,sans-serif;">1st pooling layer</span>&nbsp;&nbsp;: <br><br>the pooling layer takes patches of size [2x2] into consideration and has a stride of size = 2 as default . </h4>


```python
#adding the first pooling layer
cnn_pilot.add(MaxPooling2D(pool_size=(2,2)))

```

<h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><span style="font-size:28px;color:green;9;font-family:Monaco,sans-serif;">2nd convolutional layer</span>&nbsp;&nbsp;: <br><br> <ul><li>no of filters (n2) = 32<li>filter dimentions = [5x5]</li><li>we will use the relu function as the activation function.</li></ul>  </h4>


```python
#adding the second convolutional layer
cnn_pilot.add(Conv2D(filters = 32,
                     kernel_size = 5,
                     activation = 'relu'))
```

<h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><span style="font-size:28px;color:green;9;font-family:Monaco,sans-serif;">2nd pooling layer</span>&nbsp;&nbsp;: <br><br>the 2nd pooling has same specifications as the first pooling layer.</h4>


```python
#adding the 2nd pooling layer
cnn_pilot.add(MaxPooling2D(pool_size=(2,2)))
```

<h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><span style="font-size:28px;color:green;9;font-family:Monaco,sans-serif;">flattening layer</span>&nbsp;&nbsp;: <br><br>we will now add the flattening layer</h4>


```python
#adding the flattening layer , whose output will be used as the final NN's input
cnn_pilot.add(Flatten())
```

<h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><span style="font-size:28px;color:green;9;font-family:Monaco,sans-serif;">final neural network (prediction model):  </span><br><br><ul><li><span style="font-size:28px;color:limegreen;9;font-family:Monaco,sans-serif;">input layer :</span>&nbsp;&nbsp;the input layer for the final Neural Network is the same as the flattening layer's output, that is the flattened vector.<br><br><li><span style="font-size:28px;color:limegreen;9;font-family:Monaco,sans-serif;">hidden layer :</span>&nbsp;&nbsp;hidden layer of n3 = 64 nodes and activation funtion = relu<br><br><li><span style="font-size:28px;color:limegreen;9;font-family:Monaco,sans-serif;">output layer :</span>&nbsp;&nbsp;nodes = number of classes(10) and activation funtion = softmax </h4>


```python
#adding the hidden layer
cnn_pilot.add(Dense(units= 64,
                    activation='relu'))
cnn_pilot.add(Dense(units = num_classes,
                    activation = 'softmax'))
```

<h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:25px;color:#334761">The last thing that we need to do is 'compile' our CNN model . This is where we specify the loss-function (we will use 'categorical-crossentropy') , the optimizer (we use 'adam' so that the model's learning rate is automatically optimized ) , and for easy evaluation , the metric we will choose is accuracy .</h4>


```python
cnn_pilot.compile(loss = "categorical_crossentropy",
                   optimizer = 'adam',
                   metrics = ['accuracy'])
```

<h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:25px;color:#334761">Let us now check our final CNN model</h4>


```python
cnn_pilot.summary()
Wsave = cnn_pilot.get_weights()

```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 24, 24, 16)        416       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 12, 12, 16)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 8, 8, 32)          12832     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 4, 4, 32)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 512)               0         
    _________________________________________________________________
    dense (Dense)                (None, 64)                32832     
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                650       
    =================================================================
    Total params: 46,730
    Trainable params: 46,730
    Non-trainable params: 0
    _________________________________________________________________
    


```python
print(Wsave)
```

    [array([[[[ 0.05296947,  0.00441959, -0.08499244,  0.03550367,
              -0.00694376, -0.02069408, -0.02579778,  0.08399375,
              -0.08381316,  0.04820521,  0.02383859, -0.04553272,
               0.02609415,  0.01451211, -0.05462943, -0.07958537]],
    
            [[-0.06582969, -0.08761213, -0.08705341,  0.06112388,
              -0.0711541 , -0.05929415,  0.05159958,  0.0820345 ,
               0.108611  , -0.09432215,  0.05054851,  0.08359933,
              -0.09023547, -0.09056295,  0.05983303,  0.08756405]],
    
            [[ 0.0750574 , -0.07931793, -0.06056337,  0.02978933,
               0.07682668,  0.05830376,  0.03004429,  0.01446119,
               0.01956685, -0.00076192, -0.06267947, -0.00673611,
               0.01509555,  0.05442086, -0.08916447, -0.00804446]],
    
            [[ 0.07752787,  0.09838554,  0.1077348 , -0.1145687 ,
              -0.04359838,  0.11135407, -0.10427475, -0.03892644,
              -0.00984594, -0.04867153,  0.06120141, -0.01281781,
               0.03277589, -0.02645899, -0.04226564, -0.08395426]],
    
            [[ 0.01259033,  0.00162183, -0.11607413,  0.06845377,
               0.0545214 , -0.06697479, -0.09087063,  0.11410277,
              -0.01056775, -0.05142105,  0.02130651,  0.05106638,
              -0.10296816,  0.08964166, -0.11870342,  0.0322727 ]]],
    
    
           [[[ 0.07608002,  0.04771663,  0.11789165,  0.0093125 ,
              -0.00320292, -0.00040059,  0.07630263,  0.06741744,
               0.08772111, -0.00911827, -0.01305483,  0.11254048,
               0.01550904,  0.0311792 ,  0.08315149, -0.08063741]],
    
            [[ 0.03612505,  0.09583919, -0.07805556, -0.08411095,
               0.04680222,  0.11589572,  0.02004641,  0.02095652,
               0.09699151,  0.10538378, -0.08340237,  0.01304851,
              -0.00403651, -0.00426466,  0.03010444,  0.0504884 ]],
    
            [[ 0.09724356,  0.08014452, -0.08769351,  0.06663284,
               0.05497138,  0.01597254,  0.00524106,  0.10937439,
              -0.041958  , -0.09895582,  0.0360436 , -0.0754406 ,
              -0.11844414,  0.06205098,  0.06511168,  0.02591664]],
    
            [[-0.03876499,  0.0556571 ,  0.00440656, -0.10866799,
               0.03019765,  0.10856295,  0.09674488, -0.07144554,
               0.00632393, -0.0926316 , -0.02094746,  0.09060431,
               0.0284088 , -0.00816026, -0.02202951,  0.02883647]],
    
            [[ 0.02714293, -0.03799886, -0.05037352,  0.08606341,
               0.05695742, -0.08581673, -0.11357339,  0.03393486,
               0.04694664, -0.11503012, -0.03070076, -0.02605194,
               0.08612192,  0.00670402, -0.07206497,  0.08591713]]],
    
    
           [[[-0.04470808,  0.01619782, -0.06789019, -0.1049595 ,
              -0.06568154, -0.02397006,  0.00050122, -0.11015099,
               0.02481887,  0.07144859,  0.09468898,  0.10746792,
              -0.08000483, -0.10723815,  0.00659736, -0.00554335]],
    
            [[ 0.09285954, -0.02091116, -0.02302653,  0.02224894,
              -0.07920524, -0.03980198, -0.01478468,  0.07846281,
               0.10177879,  0.11564906,  0.02424617,  0.05945021,
               0.05560039, -0.08124782,  0.07572106,  0.04930453]],
    
            [[-0.11753998,  0.09704261,  0.04097784, -0.03170483,
               0.03721379, -0.04909904, -0.04984865,  0.09476994,
              -0.06057425, -0.06673844, -0.08929531, -0.01669761,
               0.05892444,  0.03623471, -0.11208465,  0.02791241]],
    
            [[-0.01239813,  0.11160922, -0.09573644, -0.03183004,
              -0.09401037, -0.0091574 , -0.10576924,  0.07314478,
              -0.11079919, -0.01892036,  0.03795937,  0.08934163,
               0.03333376,  0.08262178,  0.1022087 , -0.11500984]],
    
            [[-0.01269793, -0.07915668,  0.07383594,  0.06484433,
              -0.08470853, -0.04143488,  0.11794516,  0.05042247,
              -0.08647607, -0.01415024, -0.10307086,  0.08298527,
               0.08114599, -0.07674384,  0.11126162, -0.1146625 ]]],
    
    
           [[[ 0.09548554,  0.09462146,  0.03006533,  0.03269134,
              -0.02304267, -0.01480507,  0.11029616,  0.03703158,
              -0.08600886,  0.02688216, -0.06577854,  0.05927525,
              -0.03075297,  0.01521371, -0.04369248,  0.11128284]],
    
            [[ 0.04495159,  0.05691652, -0.04983384, -0.11464066,
               0.07601139,  0.00899136, -0.06647719,  0.07430398,
               0.03496791, -0.08229272, -0.04217425, -0.07683177,
              -0.00943558, -0.0106101 , -0.10305028, -0.0447576 ]],
    
            [[ 0.01120998, -0.0803431 ,  0.1001215 ,  0.1171518 ,
              -0.07806335, -0.05141289,  0.11176293,  0.0575389 ,
               0.10670702,  0.00198258,  0.01956013, -0.04974149,
              -0.01677588, -0.02308706,  0.07948121, -0.06734416]],
    
            [[-0.04686707,  0.05846795, -0.0302074 ,  0.10254088,
               0.04804462, -0.02328865, -0.11327127, -0.06659181,
               0.00412928, -0.04478224, -0.0735373 , -0.01187944,
              -0.09751697, -0.01060307, -0.07349008,  0.09869809]],
    
            [[ 0.07343246, -0.10139653, -0.08890042, -0.03669634,
               0.02059256, -0.01150323, -0.07532389, -0.10092832,
              -0.03812797, -0.01740222,  0.00224454, -0.03288941,
               0.02943978, -0.00198967,  0.09587584,  0.08112232]]],
    
    
           [[[-0.10263103,  0.04709812,  0.06297278, -0.08388063,
              -0.00588629,  0.08044706,  0.00475185,  0.04015642,
              -0.01353577, -0.07636368, -0.1073358 , -0.05524302,
              -0.07412767,  0.09964468, -0.05982959, -0.09635233]],
    
            [[-0.00095832, -0.04328419, -0.05386411, -0.01853237,
               0.07305609,  0.04865433,  0.023853  ,  0.03629184,
              -0.04513913,  0.11645257, -0.00267306, -0.07679854,
               0.05092753,  0.06817761,  0.09444927, -0.08192424]],
    
            [[-0.04509397,  0.09060335,  0.0585607 ,  0.02007939,
               0.10969606, -0.07709865, -0.04975095, -0.06869565,
              -0.08102585, -0.10151132, -0.11258949, -0.02915622,
               0.01599823, -0.10713727, -0.11764488, -0.00928088]],
    
            [[ 0.04505694,  0.06873247,  0.00624275, -0.1147061 ,
              -0.01080021,  0.06182095, -0.00050974, -0.10124801,
               0.00939575, -0.09847061,  0.00278816, -0.11563867,
              -0.07704599,  0.05848081,  0.05002721, -0.09177336]],
    
            [[-0.10040881, -0.04466876, -0.10966272, -0.10868014,
               0.03154323,  0.07845096,  0.01371016, -0.10838035,
              -0.05549464,  0.06420815, -0.02581378, -0.01758765,
              -0.10022096, -0.04968353, -0.10529621,  0.0674711 ]]]],
          dtype=float32), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          dtype=float32), array([[[[-3.08462977e-03, -6.53109476e-02,  9.85654444e-03, ...,
              -1.86660811e-02,  6.68376610e-02,  2.57860348e-02],
             [ 5.35119474e-02,  6.09558076e-03,  5.05570695e-02, ...,
              -5.78676909e-02,  4.22938615e-02,  9.75836068e-03],
             [-1.52359344e-02, -6.97660893e-02, -4.80835140e-03, ...,
               5.42935655e-02, -8.79169628e-03,  5.68700582e-03],
             ...,
             [-3.88851389e-02, -2.68793851e-03, -5.17316833e-02, ...,
               1.67736560e-02, -3.47015075e-02,  6.71706423e-02],
             [ 2.25262791e-02, -6.58437461e-02, -1.39275268e-02, ...,
              -5.77969849e-03, -6.29070774e-02, -4.14639562e-02],
             [-7.40228593e-03, -6.99180365e-02,  2.37055495e-02, ...,
              -2.09581107e-03,  1.63315535e-02,  1.09864399e-02]],
    
            [[ 5.98647073e-02,  2.70216912e-02, -5.86163737e-02, ...,
              -5.52033894e-02,  6.31891713e-02, -2.03735679e-02],
             [ 6.64450228e-04, -4.49631512e-02, -3.45084071e-02, ...,
              -1.38148256e-02,  3.94185483e-02,  2.68752128e-03],
             [-5.86503930e-02,  5.05005717e-02,  6.35649636e-02, ...,
               1.76008791e-03,  1.65244490e-02, -3.87690663e-02],
             ...,
             [ 6.67841062e-02, -4.83701527e-02, -4.56558280e-02, ...,
               3.13295275e-02, -4.89013717e-02,  2.07102373e-02],
             [ 4.73085046e-02,  3.81104946e-02,  4.23851833e-02, ...,
              -6.94756955e-03,  4.37905267e-02,  1.34190172e-03],
             [ 4.68529165e-02, -1.56870559e-02,  7.73794204e-03, ...,
               4.81239855e-02,  6.90119490e-02, -3.17590833e-02]],
    
            [[-3.34800407e-02, -2.91237757e-02, -5.55563606e-02, ...,
               6.62061945e-02,  2.31874958e-02, -1.73037276e-02],
             [-5.52598462e-02,  3.08499634e-02,  1.19088292e-02, ...,
               6.90551028e-02, -5.00218570e-02, -5.22784479e-02],
             [-6.89908862e-04,  6.81511313e-03,  6.66213855e-02, ...,
               6.51274994e-02, -7.74195790e-03, -1.86012425e-02],
             ...,
             [ 6.85057566e-02, -4.76479530e-04,  5.68958446e-02, ...,
               2.33803913e-02,  6.04011938e-02,  6.94336221e-02],
             [-4.12587151e-02, -5.95530793e-02, -2.82164551e-02, ...,
              -2.74845175e-02, -6.18633851e-02,  6.80835471e-02],
             [-5.66825755e-02,  5.13800308e-02,  3.25413942e-02, ...,
              -3.63454558e-02,  3.12325507e-02,  4.39365581e-02]],
    
            [[ 2.73286924e-02,  4.04097959e-02,  3.16480696e-02, ...,
               3.12937051e-03,  2.89879143e-02, -9.13536176e-03],
             [-5.07666245e-02, -4.65284735e-02,  2.54700333e-03, ...,
              -5.52692711e-02,  2.09344774e-02, -2.69961022e-02],
             [ 1.97337940e-02,  4.36651334e-02, -5.40572293e-02, ...,
              -5.07604033e-02,  2.72693485e-03,  2.93370411e-02],
             ...,
             [ 7.99541175e-03,  1.33230239e-02,  1.26912594e-02, ...,
               6.03315607e-02,  4.43659350e-02, -5.23219407e-02],
             [ 4.89059091e-02, -4.79879230e-03,  3.31904590e-02, ...,
              -1.74419694e-02,  6.51511475e-02, -5.72480336e-02],
             [ 5.88750839e-03,  1.75715983e-02, -7.07011074e-02, ...,
               3.16966772e-02,  8.45909119e-03,  1.11142918e-02]],
    
            [[ 5.20463735e-02,  2.22050026e-02, -3.74489278e-02, ...,
              -1.49528421e-02,  2.58589163e-02,  5.95299527e-02],
             [ 2.05994919e-02,  1.02617592e-03,  4.12614793e-02, ...,
               6.32969663e-02,  2.79452354e-02, -2.70801261e-02],
             [ 1.17119551e-02,  5.55728003e-02,  3.62278819e-02, ...,
              -6.77647516e-02,  3.57331783e-02, -5.21251671e-02],
             ...,
             [-3.57397720e-02, -6.65873215e-02,  2.58849934e-02, ...,
              -1.31563433e-02, -5.88807538e-02, -2.46576779e-02],
             [ 1.23168603e-02,  2.43328735e-02, -5.53240292e-02, ...,
              -2.11509094e-02,  1.46595016e-02, -5.64768985e-02],
             [ 6.98508844e-02, -1.32257491e-02, -1.61041953e-02, ...,
               2.93268263e-02,  6.76190853e-03,  5.40290773e-02]]],
    
    
           [[[ 3.64982784e-02, -5.10675684e-02,  5.56450561e-02, ...,
              -4.38385569e-02,  6.68639317e-02, -5.98998815e-02],
             [ 6.98965043e-03,  2.51022279e-02,  3.08345854e-02, ...,
               5.72787598e-02, -3.88392508e-02, -2.84079872e-02],
             [-1.06634907e-02,  3.08965147e-03, -5.07709235e-02, ...,
               5.83359823e-02, -2.83300355e-02,  3.91415283e-02],
             ...,
             [-5.86481355e-02, -2.93370225e-02, -2.38062106e-02, ...,
               2.34619007e-02, -4.99079563e-02,  1.05393901e-02],
             [ 5.15144765e-02,  3.33555937e-02, -2.64723003e-02, ...,
               1.50128603e-02,  5.80667630e-02, -3.12213935e-02],
             [-6.72387332e-03,  5.00952452e-03,  6.09499440e-02, ...,
              -2.46310085e-02, -4.72551659e-02,  2.02806965e-02]],
    
            [[ 5.79057410e-02, -1.73054114e-02, -5.43349944e-02, ...,
              -5.62596917e-02, -6.36439323e-02, -5.23226857e-02],
             [-3.97362188e-02, -7.81790167e-03, -2.31870711e-02, ...,
              -2.95371376e-02, -4.25631627e-02, -3.94685529e-02],
             [ 6.87264279e-02, -4.32559699e-02,  4.16562781e-02, ...,
               2.70022228e-02, -5.58390468e-02,  2.10633948e-02],
             ...,
             [ 5.70801720e-02,  6.84623942e-02,  5.70297614e-02, ...,
              -3.85295562e-02, -6.79636672e-02, -4.68839183e-02],
             [-5.51666692e-02, -1.42602026e-02,  4.59573641e-02, ...,
               2.34887600e-02, -3.18025127e-02,  5.06687909e-03],
             [-7.46171176e-03,  3.31199765e-02, -1.52591616e-03, ...,
              -6.61985278e-02,  6.08614311e-02,  9.66663659e-03]],
    
            [[-6.67375326e-02,  6.99081197e-02,  3.25280130e-02, ...,
              -5.76967113e-02, -2.72500627e-02, -1.09598041e-05],
             [-8.81315768e-03, -6.40728027e-02, -5.53504378e-03, ...,
              -4.52100672e-02, -6.29026219e-02, -4.44306694e-02],
             [-1.59854256e-02, -1.65081955e-02,  2.67877802e-02, ...,
               4.42615896e-03,  5.84893450e-02, -7.45655596e-03],
             ...,
             [-1.39442198e-02, -5.56231886e-02,  2.92061120e-02, ...,
               7.04566613e-02,  1.37280673e-03,  8.33447278e-03],
             [ 1.56912580e-02, -2.62331441e-02,  1.10837594e-02, ...,
              -4.73615304e-02, -6.58342093e-02,  2.24304348e-02],
             [ 2.21852437e-02, -4.45554592e-02, -5.02421148e-02, ...,
               5.05211428e-02, -1.42732151e-02,  5.62141314e-02]],
    
            [[-4.65479940e-02, -5.07875606e-02, -3.30471583e-02, ...,
               1.73805654e-02,  1.47988200e-02,  6.62356541e-02],
             [ 5.77211604e-02,  5.69148138e-02,  3.08038145e-02, ...,
               4.75687236e-02,  1.47685930e-02,  4.65106145e-02],
             [-5.25642857e-02, -6.97740242e-02,  1.60608664e-02, ...,
              -4.63559553e-02, -1.45013146e-02,  6.55981377e-02],
             ...,
             [ 1.20900422e-02,  9.86672938e-04,  3.94970477e-02, ...,
               6.33609220e-02,  1.90938711e-02, -3.54897566e-02],
             [-2.03941204e-02, -1.37846507e-02, -5.26834801e-02, ...,
              -4.42180820e-02, -1.26455203e-02, -4.61087041e-02],
             [ 5.58292791e-02, -5.21888286e-02, -1.05729774e-02, ...,
               4.98992428e-02,  5.88776991e-02, -9.93106142e-03]],
    
            [[ 6.80329874e-02,  3.87273729e-02,  6.79501519e-02, ...,
               4.39716727e-02,  3.65483016e-02, -5.95494546e-02],
             [-7.45689124e-03, -1.76352262e-03, -5.12750819e-02, ...,
              -6.02031872e-02, -2.16318220e-02,  2.30640918e-02],
             [ 5.45988455e-02, -2.09482983e-02,  3.30451876e-03, ...,
              -5.31448200e-02,  1.87086836e-02, -1.26694292e-02],
             ...,
             [-6.99649751e-03, -2.33535469e-03,  2.76800096e-02, ...,
              -5.46867549e-02, -7.59337842e-03,  3.74458209e-02],
             [-1.21719763e-02, -1.76303312e-02, -4.09354642e-02, ...,
              -5.62600791e-02,  5.40700555e-03, -5.58629856e-02],
             [-1.67072974e-02,  1.93629712e-02, -4.66353446e-03, ...,
              -5.42108640e-02,  4.05703038e-02,  4.83032912e-02]]],
    
    
           [[[-6.15533665e-02,  2.18868405e-02, -4.65465933e-02, ...,
              -5.25754690e-03, -5.14291711e-02, -1.49955973e-02],
             [-4.84282002e-02,  2.38965526e-02, -9.46990773e-03, ...,
               4.28482890e-02, -2.27339081e-02,  2.15001404e-03],
             [-3.32303457e-02,  3.65041196e-04, -3.81727219e-02, ...,
              -6.20815679e-02,  2.10799128e-02, -6.11364506e-02],
             ...,
             [-1.90674886e-02, -5.39610833e-02, -1.19992420e-02, ...,
               2.37555057e-03,  2.41683349e-02, -6.45637810e-02],
             [ 4.84727547e-02, -3.51193845e-02, -2.94555239e-02, ...,
               7.11179525e-03,  3.28649133e-02, -2.58993246e-02],
             [ 3.01408172e-02, -4.74010259e-02, -7.25768507e-03, ...,
               7.01377913e-02, -4.62798029e-03, -7.00525194e-02]],
    
            [[-4.59640548e-02, -1.84453279e-03, -6.71028793e-02, ...,
               3.48511487e-02, -1.88673921e-02, -1.86630636e-02],
             [-3.80515754e-02, -2.32640654e-03,  5.77451065e-02, ...,
               1.36076659e-02,  4.60052937e-02, -5.26038557e-03],
             [-3.81688289e-02,  3.09596062e-02, -6.36499226e-02, ...,
              -1.01208091e-02, -1.98088177e-02,  1.40906870e-02],
             ...,
             [ 3.12386900e-02, -5.70318177e-02,  6.35416582e-02, ...,
              -1.25832632e-02,  5.99844381e-02, -6.13678209e-02],
             [-3.71339172e-03, -2.56840885e-02, -1.70079395e-02, ...,
               3.42858061e-02,  1.84901133e-02, -2.55061425e-02],
             [-4.93894815e-02,  3.08456942e-02,  5.18287569e-02, ...,
              -5.40924817e-02, -5.06705642e-02, -2.73398012e-02]],
    
            [[-4.44466025e-02,  2.21403316e-02, -4.77475263e-02, ...,
               5.58163151e-02,  6.88130632e-02,  3.97200510e-02],
             [-2.66751312e-02,  1.06097460e-02, -3.69189568e-02, ...,
               6.48289844e-02,  2.55116746e-02,  3.35694775e-02],
             [ 1.40631422e-02, -6.28186762e-03, -3.68782058e-02, ...,
               3.55226323e-02,  8.49825144e-03, -3.84070948e-02],
             ...,
             [ 4.28815186e-02,  2.35945135e-02, -2.14903764e-02, ...,
              -6.55313581e-02, -4.23989370e-02,  1.61945373e-02],
             [ 3.78665328e-02,  5.76847270e-02,  1.42675340e-02, ...,
              -5.24931774e-02, -1.85990334e-02, -4.03915346e-02],
             [ 3.54951844e-02,  6.50333986e-02,  5.22406325e-02, ...,
              -7.03710318e-02,  5.19304797e-02,  1.11053064e-02]],
    
            [[ 5.99099621e-02,  4.38739285e-02,  6.43820837e-02, ...,
              -2.51543187e-02,  4.98059765e-02, -1.53619535e-02],
             [ 4.57044020e-02,  3.56607735e-02,  2.78698057e-02, ...,
               6.89151511e-02,  5.47023788e-02, -5.13106063e-02],
             [-6.36124238e-02,  2.80074552e-02,  4.23126221e-02, ...,
              -3.88403945e-02,  6.87986836e-02, -5.80838397e-02],
             ...,
             [ 4.89670187e-02,  2.77286470e-02,  1.18782297e-02, ...,
              -4.64421734e-02, -5.53918704e-02, -1.24639682e-02],
             [ 1.85084492e-02,  6.42084107e-02,  6.53004870e-02, ...,
              -1.99674591e-02, -4.56492007e-02, -1.49436891e-02],
             [ 2.05000415e-02,  6.92766234e-02,  6.43139854e-02, ...,
               4.03907448e-02, -2.83493884e-02,  4.47940081e-02]],
    
            [[ 1.42108724e-02,  1.19940192e-02, -2.86081694e-02, ...,
               6.17399439e-02,  2.66676471e-02, -4.15800065e-02],
             [ 2.33365968e-02,  3.54368836e-02,  6.42949566e-02, ...,
              -5.84388152e-02,  4.62596416e-02, -4.91729975e-02],
             [ 5.48207238e-02,  3.39494422e-02,  2.69616917e-02, ...,
              -3.59040499e-03,  3.76753062e-02,  6.95140138e-02],
             ...,
             [-5.91364652e-02, -2.17588842e-02, -5.94010800e-02, ...,
              -3.66884619e-02, -4.70887870e-02, -4.83488292e-02],
             [ 3.15600857e-02,  1.13205761e-02,  5.66797778e-02, ...,
              -5.08703887e-02, -5.41307852e-02,  1.53562576e-02],
             [-4.91577387e-03, -6.89216629e-02, -5.25745377e-02, ...,
              -1.42747499e-02,  3.42798010e-02,  2.77838260e-02]]],
    
    
           [[[ 5.04908115e-02,  4.56683710e-02, -4.61393893e-02, ...,
              -1.09537318e-02,  5.11954725e-03, -6.99001104e-02],
             [-3.56532671e-02,  6.21623918e-02, -4.91047055e-02, ...,
              -4.92421724e-02, -4.28224802e-02, -4.13090885e-02],
             [-1.14473887e-02,  7.01080337e-02,  5.13611436e-02, ...,
              -6.63683265e-02,  5.23470789e-02, -6.33221045e-02],
             ...,
             [ 1.96288526e-02,  5.07190302e-02,  6.92388490e-02, ...,
               6.13067225e-02,  3.75480577e-02, -8.66920128e-03],
             [ 6.10582754e-02,  1.54214799e-02, -2.62270570e-02, ...,
               2.49374658e-02, -2.77082995e-02,  5.50710931e-02],
             [-4.89291549e-02, -9.43905860e-03, -5.62075973e-02, ...,
              -7.41697848e-04, -6.00073859e-02, -2.54079439e-02]],
    
            [[ 5.50433621e-02,  1.77624673e-02,  8.48183781e-03, ...,
              -2.96118557e-02, -4.36872840e-02,  3.05946320e-02],
             [-6.38302565e-02,  4.15371209e-02,  1.28405243e-02, ...,
               4.04641479e-02,  3.26585695e-02,  2.27708444e-02],
             [ 3.89157534e-02,  1.75274238e-02, -1.76157132e-02, ...,
               3.47415283e-02, -5.83810732e-02,  6.27632961e-02],
             ...,
             [-1.17513686e-02, -4.41477150e-02,  2.31716111e-02, ...,
               4.20078486e-02,  2.65577435e-03, -1.56748854e-02],
             [ 3.33279595e-02, -2.14198045e-02,  3.20486203e-02, ...,
              -2.45996490e-02, -5.51717095e-02, -5.43309487e-02],
             [-4.66912985e-03, -4.59574163e-03,  4.57859635e-02, ...,
               1.37402937e-02, -3.68998386e-02,  6.74891844e-02]],
    
            [[-5.10874949e-02, -1.60775334e-03,  5.02412394e-02, ...,
              -3.58929485e-02, -7.03338534e-02,  1.36455521e-02],
             [-4.10981029e-02,  6.87423125e-02,  4.78965566e-02, ...,
               2.85753757e-02,  4.47238907e-02, -5.70601262e-02],
             [ 6.03998080e-02,  5.14409542e-02, -6.23879582e-03, ...,
               3.07873115e-02, -8.59554484e-03,  2.67886221e-02],
             ...,
             [ 5.50681576e-02, -7.37984478e-03, -3.60873826e-02, ...,
               6.56225309e-02, -1.34396181e-02,  6.11364841e-04],
             [-6.80764541e-02,  6.62967563e-04,  5.93346879e-02, ...,
               6.87894225e-03,  2.23917589e-02, -4.83298264e-02],
             [-6.83196932e-02, -1.59918480e-02, -5.08270934e-02, ...,
              -6.41138032e-02,  3.66273001e-02, -2.87628323e-02]],
    
            [[-3.50320078e-02,  2.99231187e-02, -1.74250267e-02, ...,
               5.03483042e-02,  1.11202076e-02, -1.59588456e-03],
             [ 4.78631929e-02,  1.57246515e-02,  5.37815019e-02, ...,
               4.74774018e-02,  4.57373708e-02, -3.90560888e-02],
             [ 6.92107454e-02,  2.13229656e-03, -3.67991216e-02, ...,
              -1.97614282e-02, -1.45882741e-02, -3.27180773e-02],
             ...,
             [-5.34684882e-02,  2.36404017e-02,  6.34326562e-02, ...,
               5.79918846e-02, -3.53398137e-02,  1.39490366e-02],
             [-6.26083389e-02,  3.00187767e-02,  4.84383479e-02, ...,
               5.63033894e-02, -5.01365289e-02,  1.00662187e-02],
             [ 6.11991659e-02,  2.00995952e-02, -5.68841547e-02, ...,
              -4.56873402e-02,  3.50890756e-02, -2.17560865e-02]],
    
            [[-3.43842432e-02, -1.13547333e-02,  4.79242876e-02, ...,
               3.25774252e-02, -4.32932787e-02,  3.95919234e-02],
             [-4.54012305e-03,  1.20633394e-02,  3.69801000e-02, ...,
               2.20878646e-02,  6.33288175e-03, -1.68531425e-02],
             [-3.46112289e-02,  5.98915145e-02,  2.65566781e-02, ...,
               1.25896186e-02,  2.45653763e-02,  2.89655924e-02],
             ...,
             [ 6.67417049e-03, -2.45461576e-02,  1.15279853e-03, ...,
              -1.31204501e-02,  1.26636103e-02, -3.67543958e-02],
             [ 1.47129446e-02,  2.80286819e-02, -6.29509762e-02, ...,
              -5.48693836e-02, -8.24449584e-03, -4.00958136e-02],
             [-1.46250427e-02,  5.93345091e-02,  8.68301094e-03, ...,
               3.40268910e-02, -3.19925435e-02,  6.05795011e-02]]],
    
    
           [[[-3.86373848e-02,  3.26821655e-02,  1.90861002e-02, ...,
               2.41560936e-02,  6.85978681e-03, -5.50094955e-02],
             [ 4.22157720e-02,  6.18200973e-02,  6.05607554e-02, ...,
               2.74304003e-02, -6.33967668e-02,  5.28428629e-02],
             [-7.04723820e-02,  4.44028527e-02,  3.65567133e-02, ...,
               7.04739168e-02, -3.25494744e-02,  3.47029045e-02],
             ...,
             [-5.21506406e-02, -5.63738234e-02,  1.68957636e-02, ...,
               6.66687414e-02, -1.51041336e-02, -3.11474502e-02],
             [ 6.76814839e-02, -6.10755086e-02, -4.10413891e-02, ...,
               1.84098110e-02, -6.74739182e-02, -1.66459829e-02],
             [ 4.13573533e-03, -3.37852202e-02,  2.62856260e-02, ...,
               5.30610234e-03, -4.12416197e-02,  4.18886244e-02]],
    
            [[ 4.50339094e-02, -1.48499869e-02,  1.54660642e-03, ...,
              -6.91713616e-02,  4.44909260e-02,  4.53367829e-04],
             [ 2.16934755e-02, -1.30311400e-03,  4.17253450e-02, ...,
               1.51444599e-02,  2.60097980e-02,  6.39722720e-02],
             [-4.92127202e-02, -4.01905105e-02, -2.60427594e-02, ...,
               5.69557920e-02, -1.75873935e-03, -3.17693017e-02],
             ...,
             [-8.55191797e-03, -4.70430478e-02,  6.58056512e-02, ...,
              -2.47043408e-02, -6.08888641e-02,  5.14582545e-03],
             [ 2.91299671e-02, -5.52596301e-02, -5.94215095e-03, ...,
              -6.09160736e-02,  6.30466267e-02, -1.54394880e-02],
             [-4.28189412e-02,  1.81024969e-02, -2.46484727e-02, ...,
              -6.97813779e-02, -4.60441709e-02,  3.97986472e-02]],
    
            [[ 4.99629527e-02,  1.15654841e-02, -1.74310952e-02, ...,
              -6.27988055e-02,  1.17515698e-02, -2.46883780e-02],
             [ 5.78719452e-02, -4.77682129e-02, -9.49374586e-03, ...,
               4.76907790e-02,  3.23824063e-02,  4.11566049e-02],
             [-5.14228158e-02, -5.73406219e-02,  4.02549803e-02, ...,
               6.01908788e-02, -6.66423962e-02,  9.64522362e-04],
             ...,
             [ 1.67738572e-02,  4.16168272e-02,  4.65576202e-02, ...,
              -1.18563809e-02, -1.34254061e-02,  4.14878279e-02],
             [ 3.66488472e-02, -2.99184136e-02,  1.99740306e-02, ...,
              -2.72844546e-02,  6.52498677e-02, -6.70640692e-02],
             [-6.50832504e-02,  3.49685475e-02,  6.01075217e-02, ...,
               6.09323159e-02,  5.68784401e-02,  6.17786497e-03]],
    
            [[-3.68881896e-02, -6.06156513e-02,  2.72297114e-02, ...,
              -2.88568512e-02,  1.46801546e-02,  6.06312677e-02],
             [ 3.97606120e-02,  1.29619092e-02,  2.29331106e-03, ...,
              -4.32111323e-03, -3.90260629e-02, -4.82981503e-02],
             [-4.93344069e-02, -3.50528620e-02, -2.13661045e-03, ...,
               1.13538653e-03, -3.45950462e-02,  1.02230534e-02],
             ...,
             [ 2.37833187e-02, -3.31788771e-02, -3.57181393e-02, ...,
              -2.53338329e-02, -1.40984580e-02,  2.82845646e-03],
             [-5.13939187e-02,  4.40067872e-02, -5.98013401e-03, ...,
              -9.58501920e-03, -1.03133507e-02,  5.02094924e-02],
             [-3.02084498e-02, -4.63533252e-02,  4.51646522e-02, ...,
              -3.54017690e-02,  3.75544131e-02,  1.74783319e-02]],
    
            [[-2.75506191e-02,  6.63413778e-02,  6.71639517e-02, ...,
               2.26892307e-02,  1.86568275e-02,  5.15624732e-02],
             [ 5.04934937e-02, -6.15476519e-02, -4.22222167e-03, ...,
               5.60989454e-02,  6.03436455e-02,  4.45485264e-02],
             [ 3.13072205e-02,  5.88920340e-02,  6.93722293e-02, ...,
               2.42237821e-02, -1.36670098e-02,  5.08258492e-02],
             ...,
             [ 3.05815488e-02, -4.03409451e-03,  6.61294237e-02, ...,
              -4.97905686e-02,  3.86222452e-02, -2.44114734e-02],
             [-2.89757922e-02, -1.79319680e-02,  3.46975103e-02, ...,
              -6.52255416e-02, -6.97524324e-02,  2.08662972e-02],
             [-4.22168300e-02, -4.82957065e-02,  6.68639168e-02, ...,
              -3.12574208e-04, -4.15507108e-02, -9.51058790e-03]]]],
          dtype=float32), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          dtype=float32), array([[ 0.06092235, -0.06739539,  0.00022747, ..., -0.03904141,
            -0.07423212,  0.02591573],
           [-0.10184709,  0.07492195, -0.09317451, ..., -0.01380756,
            -0.0869027 , -0.06285132],
           [ 0.03171752,  0.09027882,  0.04859343, ..., -0.01051477,
             0.04364237,  0.06277813],
           ...,
           [-0.06788795,  0.06615463, -0.01650497, ...,  0.06545858,
             0.02673407,  0.06954004],
           [-0.02002126,  0.08069485,  0.10037343, ..., -0.07774143,
            -0.00808933, -0.04052897],
           [-0.01522917,  0.08576541, -0.08411206, ..., -0.03738647,
            -0.10049967,  0.05560598]], dtype=float32), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32), array([[ 0.05116025, -0.22772886, -0.09936155, -0.03889425, -0.21773165,
            -0.244446  , -0.13950558, -0.13112502,  0.07116491, -0.04961558],
           [-0.22052154,  0.04413936,  0.1512047 , -0.07834195,  0.18892178,
            -0.16542313,  0.00264585, -0.0286786 , -0.17098595,  0.25518456],
           [ 0.06294417, -0.09413892, -0.08695191,  0.01751545,  0.1918869 ,
            -0.2360509 , -0.1657849 ,  0.20450944,  0.0192773 ,  0.2454367 ],
           [ 0.19260305, -0.00627431,  0.11293828, -0.05764121,  0.2781267 ,
            -0.27355927, -0.01996413,  0.04121673, -0.18965551, -0.2260243 ],
           [ 0.15453711,  0.14582172,  0.10876995, -0.00968656, -0.1338285 ,
            -0.05497874, -0.04732899,  0.01456833,  0.13461548, -0.16325733],
           [ 0.06135142,  0.21523938, -0.19517294, -0.20375729, -0.20033094,
             0.24419865, -0.26494822, -0.01899889,  0.18818763,  0.01595032],
           [-0.23500466, -0.01262528, -0.20909731, -0.10493933,  0.25101885,
             0.2252551 , -0.23056024, -0.11025837, -0.18949795,  0.1827339 ],
           [ 0.09541062, -0.02058294,  0.24906334,  0.03934565,  0.08548123,
            -0.17089567, -0.02479106,  0.01765272, -0.06134233, -0.28006163],
           [ 0.18622732, -0.23869546, -0.08059926,  0.01849923, -0.2638277 ,
            -0.26716214,  0.17467001,  0.12932685, -0.21384323,  0.0449068 ],
           [-0.06590392,  0.20926031, -0.06942669,  0.08805978,  0.26388308,
            -0.04623897,  0.01043007, -0.2641916 , -0.00188473,  0.22071555],
           [-0.10046782, -0.1785672 , -0.09324177,  0.01648951, -0.14674392,
            -0.28465316,  0.16122195, -0.20619893, -0.06704758, -0.17426065],
           [ 0.22246537, -0.14371851,  0.28404543, -0.24396473,  0.27908233,
             0.0538837 ,  0.03327638, -0.28327176, -0.08139206,  0.04631203],
           [ 0.13860014,  0.24019584, -0.15978995,  0.04415938, -0.0710603 ,
            -0.06777318, -0.16113654, -0.09083632, -0.1993164 , -0.21317723],
           [ 0.1864768 , -0.26851913,  0.23655286,  0.16540378, -0.21367969,
            -0.07161434,  0.20250025, -0.24705572,  0.15329942, -0.24899545],
           [-0.12639241,  0.08466738, -0.08446372, -0.08993244,  0.08253309,
             0.20462173, -0.2402971 ,  0.05791208, -0.14028148, -0.09202968],
           [-0.1442253 , -0.09607132, -0.07921581, -0.12395607,  0.08184171,
            -0.14769457, -0.17060801, -0.18634993, -0.13291682,  0.00969124],
           [ 0.16578197, -0.22250275,  0.24581012, -0.09530498,  0.22095463,
            -0.2810308 ,  0.06920537, -0.07903835,  0.03336129,  0.12613729],
           [-0.22974503,  0.19241473, -0.19829386, -0.2232606 , -0.20993398,
             0.12710667, -0.2641975 , -0.18287286,  0.01857173, -0.22876851],
           [-0.17301875, -0.14987816,  0.12808618, -0.13786756, -0.21091035,
            -0.22854556,  0.02401352,  0.0290904 , -0.16304076,  0.10956529],
           [ 0.01183918, -0.05632749,  0.04424724, -0.08683881, -0.13814007,
             0.14741799,  0.01641321, -0.00591421, -0.17583391,  0.09977928],
           [-0.02582541,  0.07738805,  0.00421149,  0.26293293,  0.20108974,
             0.2030943 , -0.21360861, -0.07236546,  0.06569287,  0.1999312 ],
           [-0.00704205,  0.1643601 ,  0.03732669, -0.23164564,  0.10262343,
             0.0858627 ,  0.07363942, -0.2101517 ,  0.23996392,  0.18175295],
           [ 0.06911358,  0.23917004,  0.19357726, -0.00672373,  0.01305309,
             0.23936906, -0.18134588,  0.11811355,  0.19563442,  0.20820946],
           [-0.10546316, -0.21999949, -0.04989894,  0.26801047,  0.18905821,
             0.07639584, -0.14498301,  0.0025599 ,  0.23330715,  0.09415895],
           [-0.11596864,  0.13906464, -0.2575723 , -0.07502502, -0.07857929,
             0.0840728 ,  0.23691317,  0.01075968, -0.15436515,  0.24346682],
           [-0.05291606,  0.04365158, -0.12167317,  0.22470245,  0.2682369 ,
            -0.26765022,  0.27296343, -0.16502964,  0.14515036, -0.08991982],
           [-0.21044546, -0.10793568, -0.08340003, -0.14259271, -0.20690626,
             0.2764527 , -0.17179397,  0.0732927 ,  0.03731897,  0.20464787],
           [ 0.04368085,  0.20325002,  0.00503376, -0.0358905 , -0.15080653,
             0.10980311,  0.18448317,  0.10471895, -0.10551265,  0.18249539],
           [-0.08850406,  0.21131343, -0.1764735 ,  0.25336042,  0.19508359,
            -0.1111712 ,  0.204718  ,  0.04074743, -0.03901008, -0.03419057],
           [-0.04939194,  0.1924038 ,  0.21679375,  0.12494257, -0.12144628,
             0.19343653, -0.13083988,  0.21508032,  0.08587527, -0.17120564],
           [ 0.03474665,  0.23899958,  0.10189334, -0.266116  , -0.00321549,
             0.07012349,  0.24268821,  0.04590434,  0.1954368 ,  0.14058116],
           [ 0.2845359 , -0.11737436,  0.0561797 ,  0.15176296,  0.01607734,
             0.122738  ,  0.05974624,  0.06958053, -0.07586448, -0.05503936],
           [-0.17125289, -0.1766623 ,  0.13774619,  0.20200217, -0.19881335,
             0.12503606,  0.26907197, -0.16000068,  0.04008734,  0.16921383],
           [-0.01333213, -0.25614396,  0.17801777, -0.18380246, -0.20854022,
            -0.17227605, -0.22177798, -0.04298376,  0.14196134, -0.12751576],
           [-0.12049781,  0.25063226, -0.2477583 , -0.14187586,  0.05277908,
            -0.20201817,  0.08152702, -0.21372747, -0.15705253,  0.00059342],
           [-0.02755773, -0.24236485,  0.1935421 ,  0.26065144,  0.24588355,
            -0.08455087,  0.24055013, -0.26267558, -0.25048357, -0.10406451],
           [-0.20674516, -0.05951177, -0.09616181,  0.03989249,  0.25769278,
            -0.25392425,  0.19160089,  0.10896379, -0.0951658 , -0.14570284],
           [ 0.10423484,  0.26968303,  0.05505511,  0.03875658,  0.12623498,
             0.22468749, -0.21254072, -0.25861737, -0.26920357, -0.05419734],
           [-0.2220599 , -0.24105223,  0.07992581, -0.11615847,  0.22585914,
             0.02569908, -0.05193256,  0.20157963,  0.05608723, -0.11941335],
           [ 0.21066481, -0.16696793,  0.0010097 ,  0.06230423,  0.26698962,
             0.04613253, -0.18684742,  0.17639181,  0.24986109, -0.16673237],
           [ 0.1013408 , -0.07519814, -0.19487944, -0.14478682,  0.23214802,
             0.10972327, -0.0427122 , -0.18384346,  0.09430009,  0.01241004],
           [-0.07324585, -0.02375668, -0.19649962, -0.27500224,  0.16170788,
            -0.06381437,  0.18468258,  0.00930387,  0.2169812 , -0.27485725],
           [-0.2475763 , -0.01407415, -0.20584434,  0.05972168, -0.2411749 ,
            -0.24076512, -0.25275412,  0.02162334, -0.01183233, -0.0817952 ],
           [ 0.28320011,  0.15318945, -0.16305624,  0.14585572,  0.25407907,
            -0.02767655, -0.27693996, -0.05830008, -0.18738544, -0.06663612],
           [-0.24197483,  0.21088278,  0.12055388, -0.10395099, -0.19719493,
            -0.07123403,  0.07084993,  0.05731058,  0.17701498,  0.282447  ],
           [-0.14563508, -0.21253426,  0.17419323, -0.15005304, -0.26052228,
            -0.0654714 ,  0.14907625, -0.06018679, -0.20525004,  0.19151044],
           [-0.20852874, -0.18009673, -0.12250766, -0.28167757, -0.18178901,
             0.16067454, -0.25872496,  0.07601261, -0.07444023, -0.10641706],
           [-0.2804993 , -0.06298375, -0.14753667,  0.02134073,  0.17247823,
            -0.23961827,  0.01171207, -0.22632022, -0.02646431,  0.15744227],
           [-0.19601563, -0.09113075, -0.16940373,  0.20233154, -0.06021434,
             0.08396235, -0.18961288, -0.06323649, -0.01999298, -0.08689903],
           [ 0.1259093 ,  0.05577534,  0.0681918 ,  0.27690497, -0.13240385,
             0.20117241,  0.20665446,  0.09168792, -0.07443255,  0.2609475 ],
           [ 0.02921668,  0.16916347,  0.26750323,  0.20284751, -0.00997025,
            -0.22583903, -0.2676297 , -0.07313573,  0.21846089,  0.05432984],
           [ 0.22793671, -0.17665932,  0.2647144 ,  0.1775803 , -0.14532599,
            -0.22041428, -0.20276353, -0.06746979, -0.06675491,  0.06604487],
           [ 0.2380893 , -0.18801959, -0.10360985,  0.1483581 ,  0.24583241,
             0.06802118, -0.13897741, -0.20119563, -0.16880284,  0.05361402],
           [ 0.23060414, -0.2615351 ,  0.07092547, -0.21551615,  0.1372154 ,
            -0.16517967, -0.24644431, -0.19405344,  0.27175418, -0.0775224 ],
           [ 0.15677583,  0.2711474 ,  0.173455  ,  0.06258103, -0.0881156 ,
            -0.03625548, -0.22795337,  0.03451705,  0.18210435,  0.13339117],
           [-0.00608903, -0.00306743,  0.12756899,  0.27369437, -0.18247   ,
            -0.18146864,  0.01403287,  0.10228664,  0.09258568, -0.08470654],
           [ 0.14127839,  0.08511037,  0.23461595, -0.17658293,  0.18403158,
            -0.15453106,  0.07423696, -0.14655967, -0.20248267,  0.06179318],
           [-0.11143549,  0.20961022, -0.0259603 , -0.0449983 ,  0.2599708 ,
            -0.19684109, -0.1921596 , -0.13334805,  0.05741119, -0.14818582],
           [-0.02418983, -0.03418574,  0.20565772, -0.07630752, -0.04239129,
            -0.09288046,  0.13563544,  0.00709406, -0.04613435,  0.0580487 ],
           [-0.01875952,  0.04444399,  0.15320009,  0.08616257, -0.08504748,
             0.07783577,  0.26319167,  0.23152134, -0.19049   ,  0.17757547],
           [ 0.12019202, -0.11158736, -0.01423734,  0.19366974,  0.10649568,
             0.26142982,  0.0593954 , -0.0804459 ,  0.08600023,  0.12657359],
           [-0.12083907,  0.23813489,  0.00111386,  0.2793437 ,  0.26536825,
            -0.22324757, -0.14437263, -0.01123577,  0.23944625, -0.2795249 ],
           [-0.25774676,  0.19161764, -0.15346514,  0.05151328, -0.13857272,
            -0.24031346,  0.1860635 ,  0.27952012,  0.02229619, -0.275962  ],
           [-0.04500692, -0.00619563,  0.15092024,  0.0577662 , -0.10110767,
            -0.20697531,  0.22044733, -0.09818035,  0.13652328, -0.04895888]],
          dtype=float32), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)]
    

<h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:25px;color:#334761">Looks like all the layers of the CNN are in place .<br>We have finally finished building our very own Convulution Neural Network!!  <span style='font-size:100px;'>&#127870;</span>      <br><br>
    <img src="https://media1.tenor.com/images/5f43741cbdfb89e277afbfe4fa3b0cbc/tenor.gif?itemid=10683543" width="600" height="500">
        <br><br></h4>

<h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:25px;color:#334761">Let us now fit our model to the data and see it in action.</h4>


```python
history = cnn_pilot.fit(x,
              y,
              batch_size = 100,
              epochs = 100,
              validation_split = 0.2 )

```

    Epoch 1/100
    336/336 [==============================] - 2s 5ms/step - loss: 0.3597 - accuracy: 0.8954 - val_loss: 0.1161 - val_accuracy: 0.9657
    Epoch 2/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0975 - accuracy: 0.9715 - val_loss: 0.0797 - val_accuracy: 0.9754
    Epoch 3/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0711 - accuracy: 0.9786 - val_loss: 0.0644 - val_accuracy: 0.9805
    Epoch 4/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0573 - accuracy: 0.9820 - val_loss: 0.0613 - val_accuracy: 0.9812
    Epoch 5/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0461 - accuracy: 0.9859 - val_loss: 0.0607 - val_accuracy: 0.9808
    Epoch 6/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0382 - accuracy: 0.9873 - val_loss: 0.0476 - val_accuracy: 0.9848
    Epoch 7/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0330 - accuracy: 0.9897 - val_loss: 0.0448 - val_accuracy: 0.9865
    Epoch 8/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0278 - accuracy: 0.9909 - val_loss: 0.0409 - val_accuracy: 0.9869
    Epoch 9/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0238 - accuracy: 0.9923 - val_loss: 0.0389 - val_accuracy: 0.9876
    Epoch 10/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0219 - accuracy: 0.9928 - val_loss: 0.0404 - val_accuracy: 0.9880
    Epoch 11/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0178 - accuracy: 0.9942 - val_loss: 0.0438 - val_accuracy: 0.9877
    Epoch 12/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0153 - accuracy: 0.9951 - val_loss: 0.0399 - val_accuracy: 0.9889
    Epoch 13/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0136 - accuracy: 0.9960 - val_loss: 0.0434 - val_accuracy: 0.9890
    Epoch 14/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0132 - accuracy: 0.9959 - val_loss: 0.0466 - val_accuracy: 0.9879
    Epoch 15/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0116 - accuracy: 0.9962 - val_loss: 0.0456 - val_accuracy: 0.9889
    Epoch 16/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0132 - accuracy: 0.9954 - val_loss: 0.0560 - val_accuracy: 0.9842
    Epoch 17/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0075 - accuracy: 0.9975 - val_loss: 0.0503 - val_accuracy: 0.9870
    Epoch 18/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0075 - accuracy: 0.9974 - val_loss: 0.0483 - val_accuracy: 0.9888
    Epoch 19/100
    336/336 [==============================] - 2s 4ms/step - loss: 0.0069 - accuracy: 0.9978 - val_loss: 0.0607 - val_accuracy: 0.9862
    Epoch 20/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0071 - accuracy: 0.9977 - val_loss: 0.0619 - val_accuracy: 0.9858
    Epoch 21/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0067 - accuracy: 0.9977 - val_loss: 0.0523 - val_accuracy: 0.9885
    Epoch 22/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0043 - accuracy: 0.9986 - val_loss: 0.0498 - val_accuracy: 0.9904
    Epoch 23/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0051 - accuracy: 0.9982 - val_loss: 0.0555 - val_accuracy: 0.9894
    Epoch 24/100
    336/336 [==============================] - 2s 5ms/step - loss: 0.0042 - accuracy: 0.9985 - val_loss: 0.0562 - val_accuracy: 0.9892
    Epoch 25/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0046 - accuracy: 0.9985 - val_loss: 0.0693 - val_accuracy: 0.9867
    Epoch 26/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0076 - accuracy: 0.9974 - val_loss: 0.0573 - val_accuracy: 0.9880
    Epoch 27/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0034 - accuracy: 0.9990 - val_loss: 0.0536 - val_accuracy: 0.9896
    Epoch 28/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0032 - accuracy: 0.9988 - val_loss: 0.0594 - val_accuracy: 0.9887
    Epoch 29/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0030 - accuracy: 0.9989 - val_loss: 0.0614 - val_accuracy: 0.9892
    Epoch 30/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0050 - accuracy: 0.9981 - val_loss: 0.0612 - val_accuracy: 0.9883
    Epoch 31/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0050 - accuracy: 0.9983 - val_loss: 0.0570 - val_accuracy: 0.9898
    Epoch 32/100
    336/336 [==============================] - 2s 4ms/step - loss: 0.0039 - accuracy: 0.9988 - val_loss: 0.0577 - val_accuracy: 0.9906
    Epoch 33/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0018 - accuracy: 0.9995 - val_loss: 0.0614 - val_accuracy: 0.9894
    Epoch 34/100
    336/336 [==============================] - 1s 4ms/step - loss: 4.7697e-04 - accuracy: 0.9999 - val_loss: 0.0606 - val_accuracy: 0.9907
    Epoch 35/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0021 - accuracy: 0.9994 - val_loss: 0.0585 - val_accuracy: 0.9907
    Epoch 36/100
    336/336 [==============================] - 1s 4ms/step - loss: 4.3570e-04 - accuracy: 0.9999 - val_loss: 0.0672 - val_accuracy: 0.9895
    Epoch 37/100
    336/336 [==============================] - 1s 4ms/step - loss: 2.2059e-04 - accuracy: 1.0000 - val_loss: 0.0612 - val_accuracy: 0.9910
    Epoch 38/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0072 - accuracy: 0.9981 - val_loss: 0.0938 - val_accuracy: 0.9810
    Epoch 39/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0112 - accuracy: 0.9964 - val_loss: 0.0964 - val_accuracy: 0.9844
    Epoch 40/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0020 - accuracy: 0.9994 - val_loss: 0.0673 - val_accuracy: 0.9904
    Epoch 41/100
    336/336 [==============================] - 1s 4ms/step - loss: 4.2343e-04 - accuracy: 0.9999 - val_loss: 0.0615 - val_accuracy: 0.9910
    Epoch 42/100
    336/336 [==============================] - 1s 4ms/step - loss: 7.7559e-05 - accuracy: 1.0000 - val_loss: 0.0620 - val_accuracy: 0.9905
    Epoch 43/100
    336/336 [==============================] - 1s 4ms/step - loss: 4.6520e-05 - accuracy: 1.0000 - val_loss: 0.0630 - val_accuracy: 0.9905
    Epoch 44/100
    336/336 [==============================] - 1s 4ms/step - loss: 3.5230e-05 - accuracy: 1.0000 - val_loss: 0.0640 - val_accuracy: 0.9908
    Epoch 45/100
    336/336 [==============================] - 1s 4ms/step - loss: 2.8242e-05 - accuracy: 1.0000 - val_loss: 0.0650 - val_accuracy: 0.9905
    Epoch 46/100
    336/336 [==============================] - 1s 4ms/step - loss: 2.2510e-05 - accuracy: 1.0000 - val_loss: 0.0657 - val_accuracy: 0.9905
    Epoch 47/100
    336/336 [==============================] - 1s 4ms/step - loss: 1.9290e-05 - accuracy: 1.0000 - val_loss: 0.0667 - val_accuracy: 0.9902
    Epoch 48/100
    336/336 [==============================] - 1s 4ms/step - loss: 1.5978e-05 - accuracy: 1.0000 - val_loss: 0.0675 - val_accuracy: 0.9904
    Epoch 49/100
    336/336 [==============================] - 1s 4ms/step - loss: 1.3362e-05 - accuracy: 1.0000 - val_loss: 0.0683 - val_accuracy: 0.9902
    Epoch 50/100
    336/336 [==============================] - 2s 4ms/step - loss: 1.1578e-05 - accuracy: 1.0000 - val_loss: 0.0691 - val_accuracy: 0.9902
    Epoch 51/100
    336/336 [==============================] - 1s 4ms/step - loss: 9.6275e-06 - accuracy: 1.0000 - val_loss: 0.0700 - val_accuracy: 0.9904
    Epoch 52/100
    336/336 [==============================] - 1s 4ms/step - loss: 8.2892e-06 - accuracy: 1.0000 - val_loss: 0.0706 - val_accuracy: 0.9901
    Epoch 53/100
    336/336 [==============================] - 1s 4ms/step - loss: 7.3169e-06 - accuracy: 1.0000 - val_loss: 0.0716 - val_accuracy: 0.9904
    Epoch 54/100
    336/336 [==============================] - 1s 4ms/step - loss: 6.2592e-06 - accuracy: 1.0000 - val_loss: 0.0720 - val_accuracy: 0.9905
    Epoch 55/100
    336/336 [==============================] - 1s 4ms/step - loss: 4.9594e-06 - accuracy: 1.0000 - val_loss: 0.0732 - val_accuracy: 0.9900
    Epoch 56/100
    336/336 [==============================] - 1s 4ms/step - loss: 4.4205e-06 - accuracy: 1.0000 - val_loss: 0.0741 - val_accuracy: 0.9904
    Epoch 57/100
    336/336 [==============================] - 1s 4ms/step - loss: 3.6167e-06 - accuracy: 1.0000 - val_loss: 0.0752 - val_accuracy: 0.9902
    Epoch 58/100
    336/336 [==============================] - 1s 4ms/step - loss: 3.1394e-06 - accuracy: 1.0000 - val_loss: 0.0758 - val_accuracy: 0.9902
    Epoch 59/100
    336/336 [==============================] - 1s 4ms/step - loss: 2.8109e-06 - accuracy: 1.0000 - val_loss: 0.0768 - val_accuracy: 0.9904
    Epoch 60/100
    336/336 [==============================] - 1s 4ms/step - loss: 2.2181e-06 - accuracy: 1.0000 - val_loss: 0.0775 - val_accuracy: 0.9904
    Epoch 61/100
    336/336 [==============================] - 1s 4ms/step - loss: 2.0969e-06 - accuracy: 1.0000 - val_loss: 0.0787 - val_accuracy: 0.9905
    Epoch 62/100
    336/336 [==============================] - 1s 4ms/step - loss: 1.6616e-06 - accuracy: 1.0000 - val_loss: 0.0791 - val_accuracy: 0.9906
    Epoch 63/100
    336/336 [==============================] - 1s 4ms/step - loss: 1.3584e-06 - accuracy: 1.0000 - val_loss: 0.0803 - val_accuracy: 0.9905
    Epoch 64/100
    336/336 [==============================] - 1s 4ms/step - loss: 1.3284e-06 - accuracy: 1.0000 - val_loss: 0.0812 - val_accuracy: 0.9906
    Epoch 65/100
    336/336 [==============================] - 2s 5ms/step - loss: 1.0495e-06 - accuracy: 1.0000 - val_loss: 0.0825 - val_accuracy: 0.9906
    Epoch 66/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0186 - accuracy: 0.9954 - val_loss: 0.0571 - val_accuracy: 0.9848
    Epoch 67/100
    336/336 [==============================] - 1s 4ms/step - loss: 0.0076 - accuracy: 0.9976 - val_loss: 0.0757 - val_accuracy: 0.9876
    Epoch 68/100
    336/336 [==============================] - 2s 5ms/step - loss: 0.0029 - accuracy: 0.9991 - val_loss: 0.0595 - val_accuracy: 0.9911
    Epoch 69/100
    336/336 [==============================] - 1s 4ms/step - loss: 8.7891e-04 - accuracy: 0.9997 - val_loss: 0.0606 - val_accuracy: 0.9902
    Epoch 70/100
    336/336 [==============================] - 1s 4ms/step - loss: 3.2817e-04 - accuracy: 0.9999 - val_loss: 0.0636 - val_accuracy: 0.9908
    Epoch 71/100
    336/336 [==============================] - 1s 4ms/step - loss: 1.2564e-04 - accuracy: 0.9999 - val_loss: 0.0650 - val_accuracy: 0.9910
    Epoch 72/100
    336/336 [==============================] - 1s 4ms/step - loss: 2.8610e-05 - accuracy: 1.0000 - val_loss: 0.0647 - val_accuracy: 0.9906
    Epoch 73/100
    336/336 [==============================] - 1s 4ms/step - loss: 2.0028e-05 - accuracy: 1.0000 - val_loss: 0.0651 - val_accuracy: 0.9914
    Epoch 74/100
    336/336 [==============================] - 1s 4ms/step - loss: 1.6247e-05 - accuracy: 1.0000 - val_loss: 0.0656 - val_accuracy: 0.9914
    Epoch 75/100
    336/336 [==============================] - 1s 4ms/step - loss: 1.3277e-05 - accuracy: 1.0000 - val_loss: 0.0662 - val_accuracy: 0.9912
    Epoch 76/100
    336/336 [==============================] - 1s 4ms/step - loss: 1.1297e-05 - accuracy: 1.0000 - val_loss: 0.0668 - val_accuracy: 0.9913
    Epoch 77/100
    336/336 [==============================] - 1s 4ms/step - loss: 9.5483e-06 - accuracy: 1.0000 - val_loss: 0.0675 - val_accuracy: 0.9913
    Epoch 78/100
    336/336 [==============================] - 1s 4ms/step - loss: 8.1467e-06 - accuracy: 1.0000 - val_loss: 0.0683 - val_accuracy: 0.9912
    Epoch 79/100
    336/336 [==============================] - 1s 4ms/step - loss: 6.9687e-06 - accuracy: 1.0000 - val_loss: 0.0689 - val_accuracy: 0.9914
    Epoch 80/100
    336/336 [==============================] - 1s 4ms/step - loss: 6.0256e-06 - accuracy: 1.0000 - val_loss: 0.0695 - val_accuracy: 0.9913
    Epoch 81/100
    336/336 [==============================] - 1s 4ms/step - loss: 5.1570e-06 - accuracy: 1.0000 - val_loss: 0.0704 - val_accuracy: 0.9913
    Epoch 82/100
    336/336 [==============================] - 1s 4ms/step - loss: 4.4700e-06 - accuracy: 1.0000 - val_loss: 0.0712 - val_accuracy: 0.9911
    Epoch 83/100
    336/336 [==============================] - 1s 4ms/step - loss: 3.7768e-06 - accuracy: 1.0000 - val_loss: 0.0720 - val_accuracy: 0.9913
    Epoch 84/100
    336/336 [==============================] - 1s 4ms/step - loss: 3.2182e-06 - accuracy: 1.0000 - val_loss: 0.0729 - val_accuracy: 0.9913
    Epoch 85/100
    336/336 [==============================] - 1s 4ms/step - loss: 2.7403e-06 - accuracy: 1.0000 - val_loss: 0.0739 - val_accuracy: 0.9912
    Epoch 86/100
    336/336 [==============================] - 1s 4ms/step - loss: 2.3549e-06 - accuracy: 1.0000 - val_loss: 0.0746 - val_accuracy: 0.9912
    Epoch 87/100
    336/336 [==============================] - 1s 4ms/step - loss: 2.0296e-06 - accuracy: 1.0000 - val_loss: 0.0754 - val_accuracy: 0.9912
    Epoch 88/100
    336/336 [==============================] - 1s 4ms/step - loss: 1.7146e-06 - accuracy: 1.0000 - val_loss: 0.0765 - val_accuracy: 0.9911
    Epoch 89/100
    336/336 [==============================] - 1s 4ms/step - loss: 1.4461e-06 - accuracy: 1.0000 - val_loss: 0.0774 - val_accuracy: 0.9910
    Epoch 90/100
    336/336 [==============================] - 1s 4ms/step - loss: 1.2424e-06 - accuracy: 1.0000 - val_loss: 0.0781 - val_accuracy: 0.9910
    Epoch 91/100
    336/336 [==============================] - 1s 4ms/step - loss: 1.0497e-06 - accuracy: 1.0000 - val_loss: 0.0791 - val_accuracy: 0.9910
    Epoch 92/100
    336/336 [==============================] - 1s 4ms/step - loss: 8.9076e-07 - accuracy: 1.0000 - val_loss: 0.0799 - val_accuracy: 0.9911
    Epoch 93/100
    336/336 [==============================] - 1s 4ms/step - loss: 7.7304e-07 - accuracy: 1.0000 - val_loss: 0.0807 - val_accuracy: 0.9910
    Epoch 94/100
    336/336 [==============================] - 1s 4ms/step - loss: 6.5345e-07 - accuracy: 1.0000 - val_loss: 0.0817 - val_accuracy: 0.9910
    Epoch 95/100
    336/336 [==============================] - 1s 4ms/step - loss: 5.4747e-07 - accuracy: 1.0000 - val_loss: 0.0831 - val_accuracy: 0.9910
    Epoch 96/100
    336/336 [==============================] - 1s 4ms/step - loss: 4.6682e-07 - accuracy: 1.0000 - val_loss: 0.0838 - val_accuracy: 0.9911
    Epoch 97/100
    336/336 [==============================] - 1s 4ms/step - loss: 3.9037e-07 - accuracy: 1.0000 - val_loss: 0.0843 - val_accuracy: 0.9912
    Epoch 98/100
    336/336 [==============================] - 1s 4ms/step - loss: 3.4357e-07 - accuracy: 1.0000 - val_loss: 0.0857 - val_accuracy: 0.9910
    Epoch 99/100
    336/336 [==============================] - 1s 4ms/step - loss: 2.7967e-07 - accuracy: 1.0000 - val_loss: 0.0856 - val_accuracy: 0.9911
    Epoch 100/100
    336/336 [==============================] - 1s 4ms/step - loss: 2.4617e-07 - accuracy: 1.0000 - val_loss: 0.0869 - val_accuracy: 0.9911
    


```python
print(Wsave)
```

    [array([[[[ 0.05296947,  0.00441959, -0.08499244,  0.03550367,
              -0.00694376, -0.02069408, -0.02579778,  0.08399375,
              -0.08381316,  0.04820521,  0.02383859, -0.04553272,
               0.02609415,  0.01451211, -0.05462943, -0.07958537]],
    
            [[-0.06582969, -0.08761213, -0.08705341,  0.06112388,
              -0.0711541 , -0.05929415,  0.05159958,  0.0820345 ,
               0.108611  , -0.09432215,  0.05054851,  0.08359933,
              -0.09023547, -0.09056295,  0.05983303,  0.08756405]],
    
            [[ 0.0750574 , -0.07931793, -0.06056337,  0.02978933,
               0.07682668,  0.05830376,  0.03004429,  0.01446119,
               0.01956685, -0.00076192, -0.06267947, -0.00673611,
               0.01509555,  0.05442086, -0.08916447, -0.00804446]],
    
            [[ 0.07752787,  0.09838554,  0.1077348 , -0.1145687 ,
              -0.04359838,  0.11135407, -0.10427475, -0.03892644,
              -0.00984594, -0.04867153,  0.06120141, -0.01281781,
               0.03277589, -0.02645899, -0.04226564, -0.08395426]],
    
            [[ 0.01259033,  0.00162183, -0.11607413,  0.06845377,
               0.0545214 , -0.06697479, -0.09087063,  0.11410277,
              -0.01056775, -0.05142105,  0.02130651,  0.05106638,
              -0.10296816,  0.08964166, -0.11870342,  0.0322727 ]]],
    
    
           [[[ 0.07608002,  0.04771663,  0.11789165,  0.0093125 ,
              -0.00320292, -0.00040059,  0.07630263,  0.06741744,
               0.08772111, -0.00911827, -0.01305483,  0.11254048,
               0.01550904,  0.0311792 ,  0.08315149, -0.08063741]],
    
            [[ 0.03612505,  0.09583919, -0.07805556, -0.08411095,
               0.04680222,  0.11589572,  0.02004641,  0.02095652,
               0.09699151,  0.10538378, -0.08340237,  0.01304851,
              -0.00403651, -0.00426466,  0.03010444,  0.0504884 ]],
    
            [[ 0.09724356,  0.08014452, -0.08769351,  0.06663284,
               0.05497138,  0.01597254,  0.00524106,  0.10937439,
              -0.041958  , -0.09895582,  0.0360436 , -0.0754406 ,
              -0.11844414,  0.06205098,  0.06511168,  0.02591664]],
    
            [[-0.03876499,  0.0556571 ,  0.00440656, -0.10866799,
               0.03019765,  0.10856295,  0.09674488, -0.07144554,
               0.00632393, -0.0926316 , -0.02094746,  0.09060431,
               0.0284088 , -0.00816026, -0.02202951,  0.02883647]],
    
            [[ 0.02714293, -0.03799886, -0.05037352,  0.08606341,
               0.05695742, -0.08581673, -0.11357339,  0.03393486,
               0.04694664, -0.11503012, -0.03070076, -0.02605194,
               0.08612192,  0.00670402, -0.07206497,  0.08591713]]],
    
    
           [[[-0.04470808,  0.01619782, -0.06789019, -0.1049595 ,
              -0.06568154, -0.02397006,  0.00050122, -0.11015099,
               0.02481887,  0.07144859,  0.09468898,  0.10746792,
              -0.08000483, -0.10723815,  0.00659736, -0.00554335]],
    
            [[ 0.09285954, -0.02091116, -0.02302653,  0.02224894,
              -0.07920524, -0.03980198, -0.01478468,  0.07846281,
               0.10177879,  0.11564906,  0.02424617,  0.05945021,
               0.05560039, -0.08124782,  0.07572106,  0.04930453]],
    
            [[-0.11753998,  0.09704261,  0.04097784, -0.03170483,
               0.03721379, -0.04909904, -0.04984865,  0.09476994,
              -0.06057425, -0.06673844, -0.08929531, -0.01669761,
               0.05892444,  0.03623471, -0.11208465,  0.02791241]],
    
            [[-0.01239813,  0.11160922, -0.09573644, -0.03183004,
              -0.09401037, -0.0091574 , -0.10576924,  0.07314478,
              -0.11079919, -0.01892036,  0.03795937,  0.08934163,
               0.03333376,  0.08262178,  0.1022087 , -0.11500984]],
    
            [[-0.01269793, -0.07915668,  0.07383594,  0.06484433,
              -0.08470853, -0.04143488,  0.11794516,  0.05042247,
              -0.08647607, -0.01415024, -0.10307086,  0.08298527,
               0.08114599, -0.07674384,  0.11126162, -0.1146625 ]]],
    
    
           [[[ 0.09548554,  0.09462146,  0.03006533,  0.03269134,
              -0.02304267, -0.01480507,  0.11029616,  0.03703158,
              -0.08600886,  0.02688216, -0.06577854,  0.05927525,
              -0.03075297,  0.01521371, -0.04369248,  0.11128284]],
    
            [[ 0.04495159,  0.05691652, -0.04983384, -0.11464066,
               0.07601139,  0.00899136, -0.06647719,  0.07430398,
               0.03496791, -0.08229272, -0.04217425, -0.07683177,
              -0.00943558, -0.0106101 , -0.10305028, -0.0447576 ]],
    
            [[ 0.01120998, -0.0803431 ,  0.1001215 ,  0.1171518 ,
              -0.07806335, -0.05141289,  0.11176293,  0.0575389 ,
               0.10670702,  0.00198258,  0.01956013, -0.04974149,
              -0.01677588, -0.02308706,  0.07948121, -0.06734416]],
    
            [[-0.04686707,  0.05846795, -0.0302074 ,  0.10254088,
               0.04804462, -0.02328865, -0.11327127, -0.06659181,
               0.00412928, -0.04478224, -0.0735373 , -0.01187944,
              -0.09751697, -0.01060307, -0.07349008,  0.09869809]],
    
            [[ 0.07343246, -0.10139653, -0.08890042, -0.03669634,
               0.02059256, -0.01150323, -0.07532389, -0.10092832,
              -0.03812797, -0.01740222,  0.00224454, -0.03288941,
               0.02943978, -0.00198967,  0.09587584,  0.08112232]]],
    
    
           [[[-0.10263103,  0.04709812,  0.06297278, -0.08388063,
              -0.00588629,  0.08044706,  0.00475185,  0.04015642,
              -0.01353577, -0.07636368, -0.1073358 , -0.05524302,
              -0.07412767,  0.09964468, -0.05982959, -0.09635233]],
    
            [[-0.00095832, -0.04328419, -0.05386411, -0.01853237,
               0.07305609,  0.04865433,  0.023853  ,  0.03629184,
              -0.04513913,  0.11645257, -0.00267306, -0.07679854,
               0.05092753,  0.06817761,  0.09444927, -0.08192424]],
    
            [[-0.04509397,  0.09060335,  0.0585607 ,  0.02007939,
               0.10969606, -0.07709865, -0.04975095, -0.06869565,
              -0.08102585, -0.10151132, -0.11258949, -0.02915622,
               0.01599823, -0.10713727, -0.11764488, -0.00928088]],
    
            [[ 0.04505694,  0.06873247,  0.00624275, -0.1147061 ,
              -0.01080021,  0.06182095, -0.00050974, -0.10124801,
               0.00939575, -0.09847061,  0.00278816, -0.11563867,
              -0.07704599,  0.05848081,  0.05002721, -0.09177336]],
    
            [[-0.10040881, -0.04466876, -0.10966272, -0.10868014,
               0.03154323,  0.07845096,  0.01371016, -0.10838035,
              -0.05549464,  0.06420815, -0.02581378, -0.01758765,
              -0.10022096, -0.04968353, -0.10529621,  0.0674711 ]]]],
          dtype=float32), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          dtype=float32), array([[[[-3.08462977e-03, -6.53109476e-02,  9.85654444e-03, ...,
              -1.86660811e-02,  6.68376610e-02,  2.57860348e-02],
             [ 5.35119474e-02,  6.09558076e-03,  5.05570695e-02, ...,
              -5.78676909e-02,  4.22938615e-02,  9.75836068e-03],
             [-1.52359344e-02, -6.97660893e-02, -4.80835140e-03, ...,
               5.42935655e-02, -8.79169628e-03,  5.68700582e-03],
             ...,
             [-3.88851389e-02, -2.68793851e-03, -5.17316833e-02, ...,
               1.67736560e-02, -3.47015075e-02,  6.71706423e-02],
             [ 2.25262791e-02, -6.58437461e-02, -1.39275268e-02, ...,
              -5.77969849e-03, -6.29070774e-02, -4.14639562e-02],
             [-7.40228593e-03, -6.99180365e-02,  2.37055495e-02, ...,
              -2.09581107e-03,  1.63315535e-02,  1.09864399e-02]],
    
            [[ 5.98647073e-02,  2.70216912e-02, -5.86163737e-02, ...,
              -5.52033894e-02,  6.31891713e-02, -2.03735679e-02],
             [ 6.64450228e-04, -4.49631512e-02, -3.45084071e-02, ...,
              -1.38148256e-02,  3.94185483e-02,  2.68752128e-03],
             [-5.86503930e-02,  5.05005717e-02,  6.35649636e-02, ...,
               1.76008791e-03,  1.65244490e-02, -3.87690663e-02],
             ...,
             [ 6.67841062e-02, -4.83701527e-02, -4.56558280e-02, ...,
               3.13295275e-02, -4.89013717e-02,  2.07102373e-02],
             [ 4.73085046e-02,  3.81104946e-02,  4.23851833e-02, ...,
              -6.94756955e-03,  4.37905267e-02,  1.34190172e-03],
             [ 4.68529165e-02, -1.56870559e-02,  7.73794204e-03, ...,
               4.81239855e-02,  6.90119490e-02, -3.17590833e-02]],
    
            [[-3.34800407e-02, -2.91237757e-02, -5.55563606e-02, ...,
               6.62061945e-02,  2.31874958e-02, -1.73037276e-02],
             [-5.52598462e-02,  3.08499634e-02,  1.19088292e-02, ...,
               6.90551028e-02, -5.00218570e-02, -5.22784479e-02],
             [-6.89908862e-04,  6.81511313e-03,  6.66213855e-02, ...,
               6.51274994e-02, -7.74195790e-03, -1.86012425e-02],
             ...,
             [ 6.85057566e-02, -4.76479530e-04,  5.68958446e-02, ...,
               2.33803913e-02,  6.04011938e-02,  6.94336221e-02],
             [-4.12587151e-02, -5.95530793e-02, -2.82164551e-02, ...,
              -2.74845175e-02, -6.18633851e-02,  6.80835471e-02],
             [-5.66825755e-02,  5.13800308e-02,  3.25413942e-02, ...,
              -3.63454558e-02,  3.12325507e-02,  4.39365581e-02]],
    
            [[ 2.73286924e-02,  4.04097959e-02,  3.16480696e-02, ...,
               3.12937051e-03,  2.89879143e-02, -9.13536176e-03],
             [-5.07666245e-02, -4.65284735e-02,  2.54700333e-03, ...,
              -5.52692711e-02,  2.09344774e-02, -2.69961022e-02],
             [ 1.97337940e-02,  4.36651334e-02, -5.40572293e-02, ...,
              -5.07604033e-02,  2.72693485e-03,  2.93370411e-02],
             ...,
             [ 7.99541175e-03,  1.33230239e-02,  1.26912594e-02, ...,
               6.03315607e-02,  4.43659350e-02, -5.23219407e-02],
             [ 4.89059091e-02, -4.79879230e-03,  3.31904590e-02, ...,
              -1.74419694e-02,  6.51511475e-02, -5.72480336e-02],
             [ 5.88750839e-03,  1.75715983e-02, -7.07011074e-02, ...,
               3.16966772e-02,  8.45909119e-03,  1.11142918e-02]],
    
            [[ 5.20463735e-02,  2.22050026e-02, -3.74489278e-02, ...,
              -1.49528421e-02,  2.58589163e-02,  5.95299527e-02],
             [ 2.05994919e-02,  1.02617592e-03,  4.12614793e-02, ...,
               6.32969663e-02,  2.79452354e-02, -2.70801261e-02],
             [ 1.17119551e-02,  5.55728003e-02,  3.62278819e-02, ...,
              -6.77647516e-02,  3.57331783e-02, -5.21251671e-02],
             ...,
             [-3.57397720e-02, -6.65873215e-02,  2.58849934e-02, ...,
              -1.31563433e-02, -5.88807538e-02, -2.46576779e-02],
             [ 1.23168603e-02,  2.43328735e-02, -5.53240292e-02, ...,
              -2.11509094e-02,  1.46595016e-02, -5.64768985e-02],
             [ 6.98508844e-02, -1.32257491e-02, -1.61041953e-02, ...,
               2.93268263e-02,  6.76190853e-03,  5.40290773e-02]]],
    
    
           [[[ 3.64982784e-02, -5.10675684e-02,  5.56450561e-02, ...,
              -4.38385569e-02,  6.68639317e-02, -5.98998815e-02],
             [ 6.98965043e-03,  2.51022279e-02,  3.08345854e-02, ...,
               5.72787598e-02, -3.88392508e-02, -2.84079872e-02],
             [-1.06634907e-02,  3.08965147e-03, -5.07709235e-02, ...,
               5.83359823e-02, -2.83300355e-02,  3.91415283e-02],
             ...,
             [-5.86481355e-02, -2.93370225e-02, -2.38062106e-02, ...,
               2.34619007e-02, -4.99079563e-02,  1.05393901e-02],
             [ 5.15144765e-02,  3.33555937e-02, -2.64723003e-02, ...,
               1.50128603e-02,  5.80667630e-02, -3.12213935e-02],
             [-6.72387332e-03,  5.00952452e-03,  6.09499440e-02, ...,
              -2.46310085e-02, -4.72551659e-02,  2.02806965e-02]],
    
            [[ 5.79057410e-02, -1.73054114e-02, -5.43349944e-02, ...,
              -5.62596917e-02, -6.36439323e-02, -5.23226857e-02],
             [-3.97362188e-02, -7.81790167e-03, -2.31870711e-02, ...,
              -2.95371376e-02, -4.25631627e-02, -3.94685529e-02],
             [ 6.87264279e-02, -4.32559699e-02,  4.16562781e-02, ...,
               2.70022228e-02, -5.58390468e-02,  2.10633948e-02],
             ...,
             [ 5.70801720e-02,  6.84623942e-02,  5.70297614e-02, ...,
              -3.85295562e-02, -6.79636672e-02, -4.68839183e-02],
             [-5.51666692e-02, -1.42602026e-02,  4.59573641e-02, ...,
               2.34887600e-02, -3.18025127e-02,  5.06687909e-03],
             [-7.46171176e-03,  3.31199765e-02, -1.52591616e-03, ...,
              -6.61985278e-02,  6.08614311e-02,  9.66663659e-03]],
    
            [[-6.67375326e-02,  6.99081197e-02,  3.25280130e-02, ...,
              -5.76967113e-02, -2.72500627e-02, -1.09598041e-05],
             [-8.81315768e-03, -6.40728027e-02, -5.53504378e-03, ...,
              -4.52100672e-02, -6.29026219e-02, -4.44306694e-02],
             [-1.59854256e-02, -1.65081955e-02,  2.67877802e-02, ...,
               4.42615896e-03,  5.84893450e-02, -7.45655596e-03],
             ...,
             [-1.39442198e-02, -5.56231886e-02,  2.92061120e-02, ...,
               7.04566613e-02,  1.37280673e-03,  8.33447278e-03],
             [ 1.56912580e-02, -2.62331441e-02,  1.10837594e-02, ...,
              -4.73615304e-02, -6.58342093e-02,  2.24304348e-02],
             [ 2.21852437e-02, -4.45554592e-02, -5.02421148e-02, ...,
               5.05211428e-02, -1.42732151e-02,  5.62141314e-02]],
    
            [[-4.65479940e-02, -5.07875606e-02, -3.30471583e-02, ...,
               1.73805654e-02,  1.47988200e-02,  6.62356541e-02],
             [ 5.77211604e-02,  5.69148138e-02,  3.08038145e-02, ...,
               4.75687236e-02,  1.47685930e-02,  4.65106145e-02],
             [-5.25642857e-02, -6.97740242e-02,  1.60608664e-02, ...,
              -4.63559553e-02, -1.45013146e-02,  6.55981377e-02],
             ...,
             [ 1.20900422e-02,  9.86672938e-04,  3.94970477e-02, ...,
               6.33609220e-02,  1.90938711e-02, -3.54897566e-02],
             [-2.03941204e-02, -1.37846507e-02, -5.26834801e-02, ...,
              -4.42180820e-02, -1.26455203e-02, -4.61087041e-02],
             [ 5.58292791e-02, -5.21888286e-02, -1.05729774e-02, ...,
               4.98992428e-02,  5.88776991e-02, -9.93106142e-03]],
    
            [[ 6.80329874e-02,  3.87273729e-02,  6.79501519e-02, ...,
               4.39716727e-02,  3.65483016e-02, -5.95494546e-02],
             [-7.45689124e-03, -1.76352262e-03, -5.12750819e-02, ...,
              -6.02031872e-02, -2.16318220e-02,  2.30640918e-02],
             [ 5.45988455e-02, -2.09482983e-02,  3.30451876e-03, ...,
              -5.31448200e-02,  1.87086836e-02, -1.26694292e-02],
             ...,
             [-6.99649751e-03, -2.33535469e-03,  2.76800096e-02, ...,
              -5.46867549e-02, -7.59337842e-03,  3.74458209e-02],
             [-1.21719763e-02, -1.76303312e-02, -4.09354642e-02, ...,
              -5.62600791e-02,  5.40700555e-03, -5.58629856e-02],
             [-1.67072974e-02,  1.93629712e-02, -4.66353446e-03, ...,
              -5.42108640e-02,  4.05703038e-02,  4.83032912e-02]]],
    
    
           [[[-6.15533665e-02,  2.18868405e-02, -4.65465933e-02, ...,
              -5.25754690e-03, -5.14291711e-02, -1.49955973e-02],
             [-4.84282002e-02,  2.38965526e-02, -9.46990773e-03, ...,
               4.28482890e-02, -2.27339081e-02,  2.15001404e-03],
             [-3.32303457e-02,  3.65041196e-04, -3.81727219e-02, ...,
              -6.20815679e-02,  2.10799128e-02, -6.11364506e-02],
             ...,
             [-1.90674886e-02, -5.39610833e-02, -1.19992420e-02, ...,
               2.37555057e-03,  2.41683349e-02, -6.45637810e-02],
             [ 4.84727547e-02, -3.51193845e-02, -2.94555239e-02, ...,
               7.11179525e-03,  3.28649133e-02, -2.58993246e-02],
             [ 3.01408172e-02, -4.74010259e-02, -7.25768507e-03, ...,
               7.01377913e-02, -4.62798029e-03, -7.00525194e-02]],
    
            [[-4.59640548e-02, -1.84453279e-03, -6.71028793e-02, ...,
               3.48511487e-02, -1.88673921e-02, -1.86630636e-02],
             [-3.80515754e-02, -2.32640654e-03,  5.77451065e-02, ...,
               1.36076659e-02,  4.60052937e-02, -5.26038557e-03],
             [-3.81688289e-02,  3.09596062e-02, -6.36499226e-02, ...,
              -1.01208091e-02, -1.98088177e-02,  1.40906870e-02],
             ...,
             [ 3.12386900e-02, -5.70318177e-02,  6.35416582e-02, ...,
              -1.25832632e-02,  5.99844381e-02, -6.13678209e-02],
             [-3.71339172e-03, -2.56840885e-02, -1.70079395e-02, ...,
               3.42858061e-02,  1.84901133e-02, -2.55061425e-02],
             [-4.93894815e-02,  3.08456942e-02,  5.18287569e-02, ...,
              -5.40924817e-02, -5.06705642e-02, -2.73398012e-02]],
    
            [[-4.44466025e-02,  2.21403316e-02, -4.77475263e-02, ...,
               5.58163151e-02,  6.88130632e-02,  3.97200510e-02],
             [-2.66751312e-02,  1.06097460e-02, -3.69189568e-02, ...,
               6.48289844e-02,  2.55116746e-02,  3.35694775e-02],
             [ 1.40631422e-02, -6.28186762e-03, -3.68782058e-02, ...,
               3.55226323e-02,  8.49825144e-03, -3.84070948e-02],
             ...,
             [ 4.28815186e-02,  2.35945135e-02, -2.14903764e-02, ...,
              -6.55313581e-02, -4.23989370e-02,  1.61945373e-02],
             [ 3.78665328e-02,  5.76847270e-02,  1.42675340e-02, ...,
              -5.24931774e-02, -1.85990334e-02, -4.03915346e-02],
             [ 3.54951844e-02,  6.50333986e-02,  5.22406325e-02, ...,
              -7.03710318e-02,  5.19304797e-02,  1.11053064e-02]],
    
            [[ 5.99099621e-02,  4.38739285e-02,  6.43820837e-02, ...,
              -2.51543187e-02,  4.98059765e-02, -1.53619535e-02],
             [ 4.57044020e-02,  3.56607735e-02,  2.78698057e-02, ...,
               6.89151511e-02,  5.47023788e-02, -5.13106063e-02],
             [-6.36124238e-02,  2.80074552e-02,  4.23126221e-02, ...,
              -3.88403945e-02,  6.87986836e-02, -5.80838397e-02],
             ...,
             [ 4.89670187e-02,  2.77286470e-02,  1.18782297e-02, ...,
              -4.64421734e-02, -5.53918704e-02, -1.24639682e-02],
             [ 1.85084492e-02,  6.42084107e-02,  6.53004870e-02, ...,
              -1.99674591e-02, -4.56492007e-02, -1.49436891e-02],
             [ 2.05000415e-02,  6.92766234e-02,  6.43139854e-02, ...,
               4.03907448e-02, -2.83493884e-02,  4.47940081e-02]],
    
            [[ 1.42108724e-02,  1.19940192e-02, -2.86081694e-02, ...,
               6.17399439e-02,  2.66676471e-02, -4.15800065e-02],
             [ 2.33365968e-02,  3.54368836e-02,  6.42949566e-02, ...,
              -5.84388152e-02,  4.62596416e-02, -4.91729975e-02],
             [ 5.48207238e-02,  3.39494422e-02,  2.69616917e-02, ...,
              -3.59040499e-03,  3.76753062e-02,  6.95140138e-02],
             ...,
             [-5.91364652e-02, -2.17588842e-02, -5.94010800e-02, ...,
              -3.66884619e-02, -4.70887870e-02, -4.83488292e-02],
             [ 3.15600857e-02,  1.13205761e-02,  5.66797778e-02, ...,
              -5.08703887e-02, -5.41307852e-02,  1.53562576e-02],
             [-4.91577387e-03, -6.89216629e-02, -5.25745377e-02, ...,
              -1.42747499e-02,  3.42798010e-02,  2.77838260e-02]]],
    
    
           [[[ 5.04908115e-02,  4.56683710e-02, -4.61393893e-02, ...,
              -1.09537318e-02,  5.11954725e-03, -6.99001104e-02],
             [-3.56532671e-02,  6.21623918e-02, -4.91047055e-02, ...,
              -4.92421724e-02, -4.28224802e-02, -4.13090885e-02],
             [-1.14473887e-02,  7.01080337e-02,  5.13611436e-02, ...,
              -6.63683265e-02,  5.23470789e-02, -6.33221045e-02],
             ...,
             [ 1.96288526e-02,  5.07190302e-02,  6.92388490e-02, ...,
               6.13067225e-02,  3.75480577e-02, -8.66920128e-03],
             [ 6.10582754e-02,  1.54214799e-02, -2.62270570e-02, ...,
               2.49374658e-02, -2.77082995e-02,  5.50710931e-02],
             [-4.89291549e-02, -9.43905860e-03, -5.62075973e-02, ...,
              -7.41697848e-04, -6.00073859e-02, -2.54079439e-02]],
    
            [[ 5.50433621e-02,  1.77624673e-02,  8.48183781e-03, ...,
              -2.96118557e-02, -4.36872840e-02,  3.05946320e-02],
             [-6.38302565e-02,  4.15371209e-02,  1.28405243e-02, ...,
               4.04641479e-02,  3.26585695e-02,  2.27708444e-02],
             [ 3.89157534e-02,  1.75274238e-02, -1.76157132e-02, ...,
               3.47415283e-02, -5.83810732e-02,  6.27632961e-02],
             ...,
             [-1.17513686e-02, -4.41477150e-02,  2.31716111e-02, ...,
               4.20078486e-02,  2.65577435e-03, -1.56748854e-02],
             [ 3.33279595e-02, -2.14198045e-02,  3.20486203e-02, ...,
              -2.45996490e-02, -5.51717095e-02, -5.43309487e-02],
             [-4.66912985e-03, -4.59574163e-03,  4.57859635e-02, ...,
               1.37402937e-02, -3.68998386e-02,  6.74891844e-02]],
    
            [[-5.10874949e-02, -1.60775334e-03,  5.02412394e-02, ...,
              -3.58929485e-02, -7.03338534e-02,  1.36455521e-02],
             [-4.10981029e-02,  6.87423125e-02,  4.78965566e-02, ...,
               2.85753757e-02,  4.47238907e-02, -5.70601262e-02],
             [ 6.03998080e-02,  5.14409542e-02, -6.23879582e-03, ...,
               3.07873115e-02, -8.59554484e-03,  2.67886221e-02],
             ...,
             [ 5.50681576e-02, -7.37984478e-03, -3.60873826e-02, ...,
               6.56225309e-02, -1.34396181e-02,  6.11364841e-04],
             [-6.80764541e-02,  6.62967563e-04,  5.93346879e-02, ...,
               6.87894225e-03,  2.23917589e-02, -4.83298264e-02],
             [-6.83196932e-02, -1.59918480e-02, -5.08270934e-02, ...,
              -6.41138032e-02,  3.66273001e-02, -2.87628323e-02]],
    
            [[-3.50320078e-02,  2.99231187e-02, -1.74250267e-02, ...,
               5.03483042e-02,  1.11202076e-02, -1.59588456e-03],
             [ 4.78631929e-02,  1.57246515e-02,  5.37815019e-02, ...,
               4.74774018e-02,  4.57373708e-02, -3.90560888e-02],
             [ 6.92107454e-02,  2.13229656e-03, -3.67991216e-02, ...,
              -1.97614282e-02, -1.45882741e-02, -3.27180773e-02],
             ...,
             [-5.34684882e-02,  2.36404017e-02,  6.34326562e-02, ...,
               5.79918846e-02, -3.53398137e-02,  1.39490366e-02],
             [-6.26083389e-02,  3.00187767e-02,  4.84383479e-02, ...,
               5.63033894e-02, -5.01365289e-02,  1.00662187e-02],
             [ 6.11991659e-02,  2.00995952e-02, -5.68841547e-02, ...,
              -4.56873402e-02,  3.50890756e-02, -2.17560865e-02]],
    
            [[-3.43842432e-02, -1.13547333e-02,  4.79242876e-02, ...,
               3.25774252e-02, -4.32932787e-02,  3.95919234e-02],
             [-4.54012305e-03,  1.20633394e-02,  3.69801000e-02, ...,
               2.20878646e-02,  6.33288175e-03, -1.68531425e-02],
             [-3.46112289e-02,  5.98915145e-02,  2.65566781e-02, ...,
               1.25896186e-02,  2.45653763e-02,  2.89655924e-02],
             ...,
             [ 6.67417049e-03, -2.45461576e-02,  1.15279853e-03, ...,
              -1.31204501e-02,  1.26636103e-02, -3.67543958e-02],
             [ 1.47129446e-02,  2.80286819e-02, -6.29509762e-02, ...,
              -5.48693836e-02, -8.24449584e-03, -4.00958136e-02],
             [-1.46250427e-02,  5.93345091e-02,  8.68301094e-03, ...,
               3.40268910e-02, -3.19925435e-02,  6.05795011e-02]]],
    
    
           [[[-3.86373848e-02,  3.26821655e-02,  1.90861002e-02, ...,
               2.41560936e-02,  6.85978681e-03, -5.50094955e-02],
             [ 4.22157720e-02,  6.18200973e-02,  6.05607554e-02, ...,
               2.74304003e-02, -6.33967668e-02,  5.28428629e-02],
             [-7.04723820e-02,  4.44028527e-02,  3.65567133e-02, ...,
               7.04739168e-02, -3.25494744e-02,  3.47029045e-02],
             ...,
             [-5.21506406e-02, -5.63738234e-02,  1.68957636e-02, ...,
               6.66687414e-02, -1.51041336e-02, -3.11474502e-02],
             [ 6.76814839e-02, -6.10755086e-02, -4.10413891e-02, ...,
               1.84098110e-02, -6.74739182e-02, -1.66459829e-02],
             [ 4.13573533e-03, -3.37852202e-02,  2.62856260e-02, ...,
               5.30610234e-03, -4.12416197e-02,  4.18886244e-02]],
    
            [[ 4.50339094e-02, -1.48499869e-02,  1.54660642e-03, ...,
              -6.91713616e-02,  4.44909260e-02,  4.53367829e-04],
             [ 2.16934755e-02, -1.30311400e-03,  4.17253450e-02, ...,
               1.51444599e-02,  2.60097980e-02,  6.39722720e-02],
             [-4.92127202e-02, -4.01905105e-02, -2.60427594e-02, ...,
               5.69557920e-02, -1.75873935e-03, -3.17693017e-02],
             ...,
             [-8.55191797e-03, -4.70430478e-02,  6.58056512e-02, ...,
              -2.47043408e-02, -6.08888641e-02,  5.14582545e-03],
             [ 2.91299671e-02, -5.52596301e-02, -5.94215095e-03, ...,
              -6.09160736e-02,  6.30466267e-02, -1.54394880e-02],
             [-4.28189412e-02,  1.81024969e-02, -2.46484727e-02, ...,
              -6.97813779e-02, -4.60441709e-02,  3.97986472e-02]],
    
            [[ 4.99629527e-02,  1.15654841e-02, -1.74310952e-02, ...,
              -6.27988055e-02,  1.17515698e-02, -2.46883780e-02],
             [ 5.78719452e-02, -4.77682129e-02, -9.49374586e-03, ...,
               4.76907790e-02,  3.23824063e-02,  4.11566049e-02],
             [-5.14228158e-02, -5.73406219e-02,  4.02549803e-02, ...,
               6.01908788e-02, -6.66423962e-02,  9.64522362e-04],
             ...,
             [ 1.67738572e-02,  4.16168272e-02,  4.65576202e-02, ...,
              -1.18563809e-02, -1.34254061e-02,  4.14878279e-02],
             [ 3.66488472e-02, -2.99184136e-02,  1.99740306e-02, ...,
              -2.72844546e-02,  6.52498677e-02, -6.70640692e-02],
             [-6.50832504e-02,  3.49685475e-02,  6.01075217e-02, ...,
               6.09323159e-02,  5.68784401e-02,  6.17786497e-03]],
    
            [[-3.68881896e-02, -6.06156513e-02,  2.72297114e-02, ...,
              -2.88568512e-02,  1.46801546e-02,  6.06312677e-02],
             [ 3.97606120e-02,  1.29619092e-02,  2.29331106e-03, ...,
              -4.32111323e-03, -3.90260629e-02, -4.82981503e-02],
             [-4.93344069e-02, -3.50528620e-02, -2.13661045e-03, ...,
               1.13538653e-03, -3.45950462e-02,  1.02230534e-02],
             ...,
             [ 2.37833187e-02, -3.31788771e-02, -3.57181393e-02, ...,
              -2.53338329e-02, -1.40984580e-02,  2.82845646e-03],
             [-5.13939187e-02,  4.40067872e-02, -5.98013401e-03, ...,
              -9.58501920e-03, -1.03133507e-02,  5.02094924e-02],
             [-3.02084498e-02, -4.63533252e-02,  4.51646522e-02, ...,
              -3.54017690e-02,  3.75544131e-02,  1.74783319e-02]],
    
            [[-2.75506191e-02,  6.63413778e-02,  6.71639517e-02, ...,
               2.26892307e-02,  1.86568275e-02,  5.15624732e-02],
             [ 5.04934937e-02, -6.15476519e-02, -4.22222167e-03, ...,
               5.60989454e-02,  6.03436455e-02,  4.45485264e-02],
             [ 3.13072205e-02,  5.88920340e-02,  6.93722293e-02, ...,
               2.42237821e-02, -1.36670098e-02,  5.08258492e-02],
             ...,
             [ 3.05815488e-02, -4.03409451e-03,  6.61294237e-02, ...,
              -4.97905686e-02,  3.86222452e-02, -2.44114734e-02],
             [-2.89757922e-02, -1.79319680e-02,  3.46975103e-02, ...,
              -6.52255416e-02, -6.97524324e-02,  2.08662972e-02],
             [-4.22168300e-02, -4.82957065e-02,  6.68639168e-02, ...,
              -3.12574208e-04, -4.15507108e-02, -9.51058790e-03]]]],
          dtype=float32), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          dtype=float32), array([[ 0.06092235, -0.06739539,  0.00022747, ..., -0.03904141,
            -0.07423212,  0.02591573],
           [-0.10184709,  0.07492195, -0.09317451, ..., -0.01380756,
            -0.0869027 , -0.06285132],
           [ 0.03171752,  0.09027882,  0.04859343, ..., -0.01051477,
             0.04364237,  0.06277813],
           ...,
           [-0.06788795,  0.06615463, -0.01650497, ...,  0.06545858,
             0.02673407,  0.06954004],
           [-0.02002126,  0.08069485,  0.10037343, ..., -0.07774143,
            -0.00808933, -0.04052897],
           [-0.01522917,  0.08576541, -0.08411206, ..., -0.03738647,
            -0.10049967,  0.05560598]], dtype=float32), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32), array([[ 0.05116025, -0.22772886, -0.09936155, -0.03889425, -0.21773165,
            -0.244446  , -0.13950558, -0.13112502,  0.07116491, -0.04961558],
           [-0.22052154,  0.04413936,  0.1512047 , -0.07834195,  0.18892178,
            -0.16542313,  0.00264585, -0.0286786 , -0.17098595,  0.25518456],
           [ 0.06294417, -0.09413892, -0.08695191,  0.01751545,  0.1918869 ,
            -0.2360509 , -0.1657849 ,  0.20450944,  0.0192773 ,  0.2454367 ],
           [ 0.19260305, -0.00627431,  0.11293828, -0.05764121,  0.2781267 ,
            -0.27355927, -0.01996413,  0.04121673, -0.18965551, -0.2260243 ],
           [ 0.15453711,  0.14582172,  0.10876995, -0.00968656, -0.1338285 ,
            -0.05497874, -0.04732899,  0.01456833,  0.13461548, -0.16325733],
           [ 0.06135142,  0.21523938, -0.19517294, -0.20375729, -0.20033094,
             0.24419865, -0.26494822, -0.01899889,  0.18818763,  0.01595032],
           [-0.23500466, -0.01262528, -0.20909731, -0.10493933,  0.25101885,
             0.2252551 , -0.23056024, -0.11025837, -0.18949795,  0.1827339 ],
           [ 0.09541062, -0.02058294,  0.24906334,  0.03934565,  0.08548123,
            -0.17089567, -0.02479106,  0.01765272, -0.06134233, -0.28006163],
           [ 0.18622732, -0.23869546, -0.08059926,  0.01849923, -0.2638277 ,
            -0.26716214,  0.17467001,  0.12932685, -0.21384323,  0.0449068 ],
           [-0.06590392,  0.20926031, -0.06942669,  0.08805978,  0.26388308,
            -0.04623897,  0.01043007, -0.2641916 , -0.00188473,  0.22071555],
           [-0.10046782, -0.1785672 , -0.09324177,  0.01648951, -0.14674392,
            -0.28465316,  0.16122195, -0.20619893, -0.06704758, -0.17426065],
           [ 0.22246537, -0.14371851,  0.28404543, -0.24396473,  0.27908233,
             0.0538837 ,  0.03327638, -0.28327176, -0.08139206,  0.04631203],
           [ 0.13860014,  0.24019584, -0.15978995,  0.04415938, -0.0710603 ,
            -0.06777318, -0.16113654, -0.09083632, -0.1993164 , -0.21317723],
           [ 0.1864768 , -0.26851913,  0.23655286,  0.16540378, -0.21367969,
            -0.07161434,  0.20250025, -0.24705572,  0.15329942, -0.24899545],
           [-0.12639241,  0.08466738, -0.08446372, -0.08993244,  0.08253309,
             0.20462173, -0.2402971 ,  0.05791208, -0.14028148, -0.09202968],
           [-0.1442253 , -0.09607132, -0.07921581, -0.12395607,  0.08184171,
            -0.14769457, -0.17060801, -0.18634993, -0.13291682,  0.00969124],
           [ 0.16578197, -0.22250275,  0.24581012, -0.09530498,  0.22095463,
            -0.2810308 ,  0.06920537, -0.07903835,  0.03336129,  0.12613729],
           [-0.22974503,  0.19241473, -0.19829386, -0.2232606 , -0.20993398,
             0.12710667, -0.2641975 , -0.18287286,  0.01857173, -0.22876851],
           [-0.17301875, -0.14987816,  0.12808618, -0.13786756, -0.21091035,
            -0.22854556,  0.02401352,  0.0290904 , -0.16304076,  0.10956529],
           [ 0.01183918, -0.05632749,  0.04424724, -0.08683881, -0.13814007,
             0.14741799,  0.01641321, -0.00591421, -0.17583391,  0.09977928],
           [-0.02582541,  0.07738805,  0.00421149,  0.26293293,  0.20108974,
             0.2030943 , -0.21360861, -0.07236546,  0.06569287,  0.1999312 ],
           [-0.00704205,  0.1643601 ,  0.03732669, -0.23164564,  0.10262343,
             0.0858627 ,  0.07363942, -0.2101517 ,  0.23996392,  0.18175295],
           [ 0.06911358,  0.23917004,  0.19357726, -0.00672373,  0.01305309,
             0.23936906, -0.18134588,  0.11811355,  0.19563442,  0.20820946],
           [-0.10546316, -0.21999949, -0.04989894,  0.26801047,  0.18905821,
             0.07639584, -0.14498301,  0.0025599 ,  0.23330715,  0.09415895],
           [-0.11596864,  0.13906464, -0.2575723 , -0.07502502, -0.07857929,
             0.0840728 ,  0.23691317,  0.01075968, -0.15436515,  0.24346682],
           [-0.05291606,  0.04365158, -0.12167317,  0.22470245,  0.2682369 ,
            -0.26765022,  0.27296343, -0.16502964,  0.14515036, -0.08991982],
           [-0.21044546, -0.10793568, -0.08340003, -0.14259271, -0.20690626,
             0.2764527 , -0.17179397,  0.0732927 ,  0.03731897,  0.20464787],
           [ 0.04368085,  0.20325002,  0.00503376, -0.0358905 , -0.15080653,
             0.10980311,  0.18448317,  0.10471895, -0.10551265,  0.18249539],
           [-0.08850406,  0.21131343, -0.1764735 ,  0.25336042,  0.19508359,
            -0.1111712 ,  0.204718  ,  0.04074743, -0.03901008, -0.03419057],
           [-0.04939194,  0.1924038 ,  0.21679375,  0.12494257, -0.12144628,
             0.19343653, -0.13083988,  0.21508032,  0.08587527, -0.17120564],
           [ 0.03474665,  0.23899958,  0.10189334, -0.266116  , -0.00321549,
             0.07012349,  0.24268821,  0.04590434,  0.1954368 ,  0.14058116],
           [ 0.2845359 , -0.11737436,  0.0561797 ,  0.15176296,  0.01607734,
             0.122738  ,  0.05974624,  0.06958053, -0.07586448, -0.05503936],
           [-0.17125289, -0.1766623 ,  0.13774619,  0.20200217, -0.19881335,
             0.12503606,  0.26907197, -0.16000068,  0.04008734,  0.16921383],
           [-0.01333213, -0.25614396,  0.17801777, -0.18380246, -0.20854022,
            -0.17227605, -0.22177798, -0.04298376,  0.14196134, -0.12751576],
           [-0.12049781,  0.25063226, -0.2477583 , -0.14187586,  0.05277908,
            -0.20201817,  0.08152702, -0.21372747, -0.15705253,  0.00059342],
           [-0.02755773, -0.24236485,  0.1935421 ,  0.26065144,  0.24588355,
            -0.08455087,  0.24055013, -0.26267558, -0.25048357, -0.10406451],
           [-0.20674516, -0.05951177, -0.09616181,  0.03989249,  0.25769278,
            -0.25392425,  0.19160089,  0.10896379, -0.0951658 , -0.14570284],
           [ 0.10423484,  0.26968303,  0.05505511,  0.03875658,  0.12623498,
             0.22468749, -0.21254072, -0.25861737, -0.26920357, -0.05419734],
           [-0.2220599 , -0.24105223,  0.07992581, -0.11615847,  0.22585914,
             0.02569908, -0.05193256,  0.20157963,  0.05608723, -0.11941335],
           [ 0.21066481, -0.16696793,  0.0010097 ,  0.06230423,  0.26698962,
             0.04613253, -0.18684742,  0.17639181,  0.24986109, -0.16673237],
           [ 0.1013408 , -0.07519814, -0.19487944, -0.14478682,  0.23214802,
             0.10972327, -0.0427122 , -0.18384346,  0.09430009,  0.01241004],
           [-0.07324585, -0.02375668, -0.19649962, -0.27500224,  0.16170788,
            -0.06381437,  0.18468258,  0.00930387,  0.2169812 , -0.27485725],
           [-0.2475763 , -0.01407415, -0.20584434,  0.05972168, -0.2411749 ,
            -0.24076512, -0.25275412,  0.02162334, -0.01183233, -0.0817952 ],
           [ 0.28320011,  0.15318945, -0.16305624,  0.14585572,  0.25407907,
            -0.02767655, -0.27693996, -0.05830008, -0.18738544, -0.06663612],
           [-0.24197483,  0.21088278,  0.12055388, -0.10395099, -0.19719493,
            -0.07123403,  0.07084993,  0.05731058,  0.17701498,  0.282447  ],
           [-0.14563508, -0.21253426,  0.17419323, -0.15005304, -0.26052228,
            -0.0654714 ,  0.14907625, -0.06018679, -0.20525004,  0.19151044],
           [-0.20852874, -0.18009673, -0.12250766, -0.28167757, -0.18178901,
             0.16067454, -0.25872496,  0.07601261, -0.07444023, -0.10641706],
           [-0.2804993 , -0.06298375, -0.14753667,  0.02134073,  0.17247823,
            -0.23961827,  0.01171207, -0.22632022, -0.02646431,  0.15744227],
           [-0.19601563, -0.09113075, -0.16940373,  0.20233154, -0.06021434,
             0.08396235, -0.18961288, -0.06323649, -0.01999298, -0.08689903],
           [ 0.1259093 ,  0.05577534,  0.0681918 ,  0.27690497, -0.13240385,
             0.20117241,  0.20665446,  0.09168792, -0.07443255,  0.2609475 ],
           [ 0.02921668,  0.16916347,  0.26750323,  0.20284751, -0.00997025,
            -0.22583903, -0.2676297 , -0.07313573,  0.21846089,  0.05432984],
           [ 0.22793671, -0.17665932,  0.2647144 ,  0.1775803 , -0.14532599,
            -0.22041428, -0.20276353, -0.06746979, -0.06675491,  0.06604487],
           [ 0.2380893 , -0.18801959, -0.10360985,  0.1483581 ,  0.24583241,
             0.06802118, -0.13897741, -0.20119563, -0.16880284,  0.05361402],
           [ 0.23060414, -0.2615351 ,  0.07092547, -0.21551615,  0.1372154 ,
            -0.16517967, -0.24644431, -0.19405344,  0.27175418, -0.0775224 ],
           [ 0.15677583,  0.2711474 ,  0.173455  ,  0.06258103, -0.0881156 ,
            -0.03625548, -0.22795337,  0.03451705,  0.18210435,  0.13339117],
           [-0.00608903, -0.00306743,  0.12756899,  0.27369437, -0.18247   ,
            -0.18146864,  0.01403287,  0.10228664,  0.09258568, -0.08470654],
           [ 0.14127839,  0.08511037,  0.23461595, -0.17658293,  0.18403158,
            -0.15453106,  0.07423696, -0.14655967, -0.20248267,  0.06179318],
           [-0.11143549,  0.20961022, -0.0259603 , -0.0449983 ,  0.2599708 ,
            -0.19684109, -0.1921596 , -0.13334805,  0.05741119, -0.14818582],
           [-0.02418983, -0.03418574,  0.20565772, -0.07630752, -0.04239129,
            -0.09288046,  0.13563544,  0.00709406, -0.04613435,  0.0580487 ],
           [-0.01875952,  0.04444399,  0.15320009,  0.08616257, -0.08504748,
             0.07783577,  0.26319167,  0.23152134, -0.19049   ,  0.17757547],
           [ 0.12019202, -0.11158736, -0.01423734,  0.19366974,  0.10649568,
             0.26142982,  0.0593954 , -0.0804459 ,  0.08600023,  0.12657359],
           [-0.12083907,  0.23813489,  0.00111386,  0.2793437 ,  0.26536825,
            -0.22324757, -0.14437263, -0.01123577,  0.23944625, -0.2795249 ],
           [-0.25774676,  0.19161764, -0.15346514,  0.05151328, -0.13857272,
            -0.24031346,  0.1860635 ,  0.27952012,  0.02229619, -0.275962  ],
           [-0.04500692, -0.00619563,  0.15092024,  0.0577662 , -0.10110767,
            -0.20697531,  0.22044733, -0.09818035,  0.13652328, -0.04895888]],
          dtype=float32), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)]
    

<h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:25px;color:#334761">To better analyse our data , let us plot its performance as a function of no.of epochs.</h4>


```python
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('loss')
plt.legend()
plt.show()
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.title('accuracy')
plt.legend()
plt.show()
```


    
![png](output_41_0.png)
    



    
![png](output_41_1.png)
    


<h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:25px;color:#334761">We see that after a certain no.of epochs the validation loss increases and from there on osciliates non-unformly . this means that beyond a certain epoch our model begins to overfit ! . One of the methods of controlling overfitting is by using <span style="font-size:25px;color:#334761;color:Tomato">&nbsp;&nbsp;<b><i>'early stopping' . </i></b></span><br><br><span style="font-size:25px;color:#334761;color:Tomato">&nbsp;&nbsp;<b><i>early stopping</i></b></span>&nbsp;&nbsp;: while the model trains , beyond a certain epoch the loss and accuracy tends to fluctuate. we try to 'catch' the model when it is at its best by saving the model when it has it's lowest loss or highest accuracy , some allowance is given to ensure that the we have the best model , that is called 'patience'. patience is the no of epochs that is allowed to train , to check if the accuracy/loss gets better or not .<br><br>(credits to <a href="https://www.kaggle.com/rameshbabugonegandla">@ramesh</a> for this helpfull suggestion)


```python
cnn_pilot.set_weights(Wsave)

es = EarlyStopping(monitor='val_loss', mode='min', verbose= 1 , patience=5)
mc = ModelCheckpoint("best_cnn.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)

history = cnn_pilot.fit(x,
                        y,
                        batch_size = 100,
                        epochs = 100,
                        validation_split = 0.2,
                        verbose = 1,
                        callbacks = [es,mc])
```

    Epoch 1/100
    334/336 [============================>.] - ETA: 0s - loss: 0.1875 - accuracy: 0.9418
    Epoch 00001: val_loss improved from inf to 0.06382, saving model to best_cnn.h5
    336/336 [==============================] - 1s 4ms/step - loss: 0.1868 - accuracy: 0.9421 - val_loss: 0.0638 - val_accuracy: 0.9808
    Epoch 2/100
    325/336 [============================>.] - ETA: 0s - loss: 0.0536 - accuracy: 0.9836
    Epoch 00002: val_loss improved from 0.06382 to 0.05244, saving model to best_cnn.h5
    336/336 [==============================] - 1s 4ms/step - loss: 0.0535 - accuracy: 0.9836 - val_loss: 0.0524 - val_accuracy: 0.9838
    Epoch 3/100
    331/336 [============================>.] - ETA: 0s - loss: 0.0367 - accuracy: 0.9887
    Epoch 00003: val_loss improved from 0.05244 to 0.04250, saving model to best_cnn.h5
    336/336 [==============================] - 1s 4ms/step - loss: 0.0366 - accuracy: 0.9887 - val_loss: 0.0425 - val_accuracy: 0.9865
    Epoch 4/100
    328/336 [============================>.] - ETA: 0s - loss: 0.0276 - accuracy: 0.9914
    Epoch 00004: val_loss improved from 0.04250 to 0.04091, saving model to best_cnn.h5
    336/336 [==============================] - 1s 4ms/step - loss: 0.0275 - accuracy: 0.9914 - val_loss: 0.0409 - val_accuracy: 0.9882
    Epoch 5/100
    330/336 [============================>.] - ETA: 0s - loss: 0.0212 - accuracy: 0.9935
    Epoch 00005: val_loss improved from 0.04091 to 0.03967, saving model to best_cnn.h5
    336/336 [==============================] - 1s 4ms/step - loss: 0.0214 - accuracy: 0.9934 - val_loss: 0.0397 - val_accuracy: 0.9883
    Epoch 6/100
    323/336 [===========================>..] - ETA: 0s - loss: 0.0176 - accuracy: 0.9941
    Epoch 00006: val_loss improved from 0.03967 to 0.03912, saving model to best_cnn.h5
    336/336 [==============================] - 1s 4ms/step - loss: 0.0176 - accuracy: 0.9941 - val_loss: 0.0391 - val_accuracy: 0.9876
    Epoch 7/100
    336/336 [==============================] - ETA: 0s - loss: 0.0141 - accuracy: 0.9956
    Epoch 00007: val_loss improved from 0.03912 to 0.03872, saving model to best_cnn.h5
    336/336 [==============================] - 1s 4ms/step - loss: 0.0141 - accuracy: 0.9956 - val_loss: 0.0387 - val_accuracy: 0.9896
    Epoch 8/100
    334/336 [============================>.] - ETA: 0s - loss: 0.0114 - accuracy: 0.9966
    Epoch 00008: val_loss did not improve from 0.03872
    336/336 [==============================] - 1s 4ms/step - loss: 0.0114 - accuracy: 0.9966 - val_loss: 0.0410 - val_accuracy: 0.9882
    Epoch 9/100
    328/336 [============================>.] - ETA: 0s - loss: 0.0100 - accuracy: 0.9969
    Epoch 00009: val_loss improved from 0.03872 to 0.03595, saving model to best_cnn.h5
    336/336 [==============================] - 1s 4ms/step - loss: 0.0102 - accuracy: 0.9968 - val_loss: 0.0360 - val_accuracy: 0.9904
    Epoch 10/100
    331/336 [============================>.] - ETA: 0s - loss: 0.0082 - accuracy: 0.9975
    Epoch 00010: val_loss did not improve from 0.03595
    336/336 [==============================] - 2s 5ms/step - loss: 0.0082 - accuracy: 0.9975 - val_loss: 0.0428 - val_accuracy: 0.9888
    Epoch 11/100
    333/336 [============================>.] - ETA: 0s - loss: 0.0066 - accuracy: 0.9981
    Epoch 00011: val_loss did not improve from 0.03595
    336/336 [==============================] - 1s 4ms/step - loss: 0.0067 - accuracy: 0.9981 - val_loss: 0.0550 - val_accuracy: 0.9862
    Epoch 12/100
    334/336 [============================>.] - ETA: 0s - loss: 0.0065 - accuracy: 0.9979
    Epoch 00012: val_loss did not improve from 0.03595
    336/336 [==============================] - 1s 4ms/step - loss: 0.0065 - accuracy: 0.9979 - val_loss: 0.0423 - val_accuracy: 0.9900
    Epoch 13/100
    329/336 [============================>.] - ETA: 0s - loss: 0.0057 - accuracy: 0.9983
    Epoch 00013: val_loss did not improve from 0.03595
    336/336 [==============================] - 1s 4ms/step - loss: 0.0056 - accuracy: 0.9982 - val_loss: 0.0482 - val_accuracy: 0.9888
    Epoch 14/100
    326/336 [============================>.] - ETA: 0s - loss: 0.0048 - accuracy: 0.9985
    Epoch 00014: val_loss did not improve from 0.03595
    336/336 [==============================] - 1s 4ms/step - loss: 0.0049 - accuracy: 0.9985 - val_loss: 0.0466 - val_accuracy: 0.9886
    Epoch 00014: early stopping
    


```python
cnn_best = load_model("best_cnn.h5")
_, train_acc = cnn_best.evaluate(x, y, verbose=0)
print("Train: {}".format(train_acc))
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('loss')
plt.legend()
plt.show()
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.title('accuracy')
plt.legend()
plt.show()
```

    Train: 0.996833324432373
    


    
![png](output_44_1.png)
    



    
![png](output_44_2.png)
    


<h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:25px;color:#334761"> we see that we caught the model before it tends to overfit.

<h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:25px;color:#334761">the working of a CNN might seem a little 'blackboxy' , so to better our intuition we will visualize the journey of an image through the CNN . enjoy the ride ;)</h4>


```python
def image_journey(img_num):
    layer_outputs = [layer.output for layer in cnn_pilot.layers[:4]]
    activation_model = keras.Model(inputs=cnn_pilot.inputs, outputs=layer_outputs)
    activations = activation_model.predict(x[img_num].reshape(1,28,28,1))
    layer_list=['1st convolution layer','1st pooling layer','2nd convolution layer','2nd pooling layer']
    fig, ax = plt.subplots(1, 1 ,figsize = (4,4))
    plt.subplots_adjust(left=0, bottom=-0.2, right=1, top=0.9,wspace=None, hspace=0.1)
    fig.suptitle('original image', fontsize=30)
    ax.imshow(x[img_num][:,:,0], cmap='gray')
    for i in range(4):
        activation = activations[i]
        activation_index=0
        fig, ax = plt.subplots(1, 6 ,figsize = (30,3))
        fig.suptitle(layer_list[i], fontsize=50)
        plt.subplots_adjust(left=0, bottom=-0.6, right=1, top=0.6,wspace=None, hspace=0.1)
        for row in range(0,6):
            ax[row].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1
image_journey(200)
```


    
![png](output_47_0.png)
    



    
![png](output_47_1.png)
    



    
![png](output_47_2.png)
    



    
![png](output_47_3.png)
    



    
![png](output_47_4.png)
    


<h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:25px;color:#334761">we finally reached the last part of the notebook , where we will submit our predictions of the test data . <br><br>Just as we pre processed our training data we will preprocess the test data , make our predictions with our CNN model and submit our predictions.</h4>


```python
mnist_test = "/kaggle/input/digit-recognizer/test.csv"
mnist_test = np.loadtxt(mnist_test, skiprows=1, delimiter=',')
num_images = mnist_test.shape[0]
out_x = mnist_test.reshape(num_images, img_rows, img_cols, 1)
out_x = out_x / 255
results = cnn_pilot.predict(out_x)
results = np.argmax(results,axis = 1)
submissions=pd.DataFrame({"ImageId": list(range(1,len(results)+1)),"Label": results})
submissions.to_csv("submission.csv", index=False, header=True)
```

<h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:25px;color:#334761"><ul><li>Thank you for reading till the end of this notebook .</li><li>Self learning is a very frustrating process , so my agenda is to make fellow learner's journey as easy as possible through my notebooks . </li><li>if you liked this notebook , please <span style="font-size:28px;color:orange;9;font-family:Monaco,sans-serif;">upvote</span>&nbsp;it to help me upload more engaging and intuitive notebooks . </li><li>cheers! and happy machine learing.<span style='font-size:80px;'>&#127867;</span></h4>


```python

```
