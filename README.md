# ChestMultiVision
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([URL_TO_YOUR_APP](https://chestmultivision.streamlit.app/))

<!-- LICENSE FOR THE README TEMPLATE USED FROM https://github.com/othneildrew/Best-README-Template

MIT License

Copyright (c) 2021 Othneil Drew

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.-->

<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Vindhyaa-Saravanan/final-year-project-Vindhyaa-Saravanan">
    <img src="app\logo.jpg" alt="Logo" width="100" height="100">
  </a>

<h3 align="center">Enhancing Medical Diagnostics with Multi-Label Classification of Chest X-rays</h3>

  <p align="center">
    Leveraging deep learning neural networks for multi-label classification of medical images, enabling early detection of multiple findings in chest X-rays.

    ChestMultiVision harnesses a custom deep learning model based on the ResNet50V2 architecture. It was trained on the Chest X-ray-14 dataset. It predicts six different findings detectable on chest x-rays, that are: Atelectasis, Effusion, Infiltration, Mass, No Finding, and Nodule.

Product Disclaimer: ChestMultiVision is a prototype chest x ray classification app, it is NOT A MEDICAL DEVICE. Predictions made are simply to demonstrate the application. Predictions are not expected to be highly accurate and should not be used for medical diagnosis of any kind. Predictions may not be as effective on chest x-rays of women, children and elderly adults. Please consult a medical professional for any medical advice or diagnosis
    <br />
    <a href="https://github.com/Vindhyaa-Saravanan/final-year-project-Vindhyaa-Saravanan"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Vindhyaa-Saravanan/final-year-project-Vindhyaa-Saravanan">View Demo</a>
    ·
    <a href="https://github.com/uol-feps-soc-comp3931-2324-classroom/final-year-project-Vindhyaa-Saravanan/issues">Report Bug</a>
    ·
    <a href="https://github.com/uol-feps-soc-comp3931-2324-classroom/final-year-project-Vindhyaa-Saravanan/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

My dissertation project focuses on addressing a critical challenge in the realm of medical imaging: the accurate detection of multiple findings in chest X-rays through deep learning-based multi-label classification. Existing CAD systems often struggle to effectively handle cases where multiple pathologies coexist within a single image, presenting a substantial obstacle for practical clinical use. My project seeks to bridge this gap by developing deep learning models capable of accurately classifying chest X-ray images with multiple overlapping labels.

Training with the NIH Chest X-ray-14 dataset, I aim to train and evaluate deep learning neural networks tailored to multi-label classification tasks. I also aim to investigate effect of training the model on various different subsets of data and labels, to compare model performance across training datasets in a data ablation study. The project will culminate in a user-friendly web application for clinicians, with an intuitive interface to upload chest X-ray images and receive rapid, accurate diagnostic predictions.

Through this project, I aim to contribute to medical imaging AI while empowering medical professionals with efficient and accurate diagnostic aids, to provide a valuable second opinion during diagnosis.

The dataset used for this machine learning task was downloaded from <a href="https://nihcc.app.box.com/v/ChestXray-NIHCC">this site</a>.

Citation for the dataset research paper: Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, Ronald Summers, ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases, IEEE CVPR, pp. 3462-3471, 2017

### Built With

* [![Python][Python.com]][Python-url]
* [![Streamlit][Streamlit.com]][Streamlit-url]
* [![Keras][Keras.com]][Keras-url]
* [![Tensorflow][Tensorflow.com]][Tensorflow-url]
* [![Colab][Colab.com]][Colab-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

This is an example of how you can go about setting up the project locally.
To get a local copy up and running follow these simple steps:

### Cloning the repository

1. When cloning this whole repository, which has large files such as the models, etc which will need to be tracked by Git LFS, make sure to set up Git LFS by:

```bash
> git lfs install
```
Refer to [StackOverflow thread](https://example.com) and GitHub docs for further details regarding setting up Git LFS.

2. Once Git LFS has been setup, clone this repository: 
```bash
> git clone https://github.com/Vindhyaa-Saravanan/ChestMultiVision.git
```
The `.gitattributes` file will make sure the appropriate files are cloned.

### Setting up the repository
*You should only need to do this once, cloning the repo for the first time, with proper git usage.*

1. Make sure you have installed Python version 3.10, and the PATH variable has been updated.

2. Create your virtual environment in the project-squad30 directory. As long as the environment is called "myenv", it will not be uploaded to version control (see the `gitignore` file).

```bash
> python -m venv myenv # where "myenv" is the name of the environment
# OR
> python3 -m venv myenv 
```

2. Activate the new virtual environment. 
```bash
# On Windows use the command:
> ./env/Scripts/activate
# In a Linux or Mac environment, use the command:
> ./env/scripts/activate

# To deactivate the environment later, when done with the app:
> deactivate
```

3. Install all project requirements as listed in `app/requirements.txt`.

```bash
> pip install -r requirements.txt
# OR
> pip3 install -r requirements.txt
```

4. Check the versions of Tensorflow and Keras, it is essential to use the right versions for the ML model to work. Run the following while the virtual environment is activated:
```bash
> pip show tensorflow
# Expect to see version number 2.15.0 in the output
> pip show keras
# Expect to see version number 2.15.0 in the output.

# OR respective command for pip3.
```

4. The app is now ready to run! 
```bash
> streamlit run app.py
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Vindhyaa Saravanan: sc21vs@leeds.ac.uk

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
I extend my gratitude to my project supervisor, Dr. Nishant Ravikumar, for their unwavering support and invaluable guidance throughout the duration of this project. Their expertise, encouragement, and constructive feedback have been instrumental in shaping the direction and success of this endeavor.

I am also deeply thankful to Dr. Arash Rabbani, my project assessor, for their insightful inputs and feedback during my project progress discussion. Their expertise and suggestions have greatly contributed to the refinement and improvement of this project.

I am equally grateful to my parents for their unwavering support, understanding, and encouragement throughout this journey. Their belief in my abilities and encouragement have been a constant source of motivation.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[Python.com]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://python.org
[Keras.com]: https://keras.io/
[Tensorflow.com]: https://www.tensorflow.org/
[Keras-url]:https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white
[Tensorflow-url]:https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white
[Streamlit.com]: https://streamlit.io/
[Streamlit-url]:https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white
[colab.com]: https://colab.google/
[colab-url]:https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252

