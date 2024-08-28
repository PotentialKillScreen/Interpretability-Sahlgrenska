### Background
The project was carried out by me and my colleague as part of a hackathon at our company Random Forest in 2023.

The code used to generate the heatmaps is found in "Gradient-Cam.py".

### Initial Plan
The objective was to use interpretability techniques on a Neural Network that had been 
developed for the Customer Sahlgrenska in order to predict the future development of cancer based on histopathological images. 
Neural Networks are often seen as black boxes where the decision making process is
unintelligible. This is a problem for two reasons. The first is that it makes improving the
performance of the algorithm more difficult. The second problem is situations where a black
box decision maker is unattractive to the customer, even if the performance is better
than more transparent techniques. The second problem is especially potent in critical areas,
such as self-driving cars or medical diagnosis, where the consequences of errors can be
catastrophic.
The short term potential benefit is an added insight for the future work of training the Neural
Network
The long term potential benefit is that this could be a start to acquiring an additional skillset at RF
that could be leveraged in our consulting services. Since there was next to no experience within
this field at the company, our ambition for the hackaton was
modest. The goal was to produce at least one visual of the network that we have some
interpretation of.

### Method
We decided to use the technique Gradient-Class Activation Mapping (Grad-CAM).
The first reason was that the results would be easily understood as the areas of the image
that the network used to predict a specific positive sample. The second reason was that we
believed the method would be easily implemented.
We found a guide on [medium](https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82)
and decided to follow it. We picked this particular guide as the
author had implemented the algorithm in PyTorch which is the library that 
the neural network is implemented by. Also the guide contained a lot of code that we
could use directly.

The original network uses Whole Scan Images that are very large at approximately 20 GB. Since
they where cumbersome to transfer and work with, we extracted two images that were part of a False Positive WSI.
We recreated the orginal network with its trained weights. We took the gradient of the output of the last
Convolutional layer with respect to the previous layer.

This generated heatmaps we overlayed on the orginal images to see what parts of the images where causing the network to 
output positive outputs.


### Results
The first image seemed to be interpreted as random noise based on the heatmap only being noise. We speculated that the first
image of a WSI would be at the edge of the area and not contain any relevant information. 

<figure>
  <img
  src="https://github.com/PotentialKillScreen/Interpretability-Sahlgrenska/blob/main/HeatMap_1.png"
  alt="Alt text">
  <figcaption>Heatmap for image 1</figcaption>
</figure>

---

The second image showed certain parts of the image being more important than others. However, these parts of the image did not stand
out to our human perception. However, neither of us has any knowledge of what to look for and we did not have time to consult with 
someone with the relevant medical knowledge.


<table>
  <tr>
    <td>
      <img src="https://github.com/PotentialKillScreen/Interpretability-Sahlgrenska/blob/main/Input_image_2.png" alt="Input Image 2" style="width: 300px;"/>
      <br>
      <figcaption>Input Image 2</figcaption>
    </td>
    <td>
      <img src="https://github.com/PotentialKillScreen/Interpretability-Sahlgrenska/blob/main/Heatmap_2.png" alt="Input Image 2" style="width: 300px;"/>
      <br>
      <figcaption>Heat Map 2</figcaption>
    </td>
    <td>
      <img src="https://github.com/PotentialKillScreen/Interpretability-Sahlgrenska/blob/main/Combined_2.png" alt="Input Image 2" style="width: 300px;"/>
      <br>
      <figcaption>Combined Image and Heatmap</figcaption>
    </td>
  </tr>
</table>

The code used to generate the heatmaps is found in "Gradient-Cam.py".

### Conclussions
We would need to continue to the project in order to derive insights usable for the continued
development of the network. Such steps could be
- Test with more images to see patterns for which images produced noisy heatmaps.
- Get input from a medical professional on the difference between the images and the heatmaps.

Currently the benefits of the projects is limited to the experience we got from reading about grad-cam 
and implementing it.
