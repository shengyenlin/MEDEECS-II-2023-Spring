# Special Topic in innovative integration of medicine and EECS (II)

This project aims to harness the power of deep learning to improve the diagnostic accuracy of hydroxychloroquine (HCQ) retinopathy through the analysis of retinal OCT B-scan images. Hydroxychloroquine, commonly used in the treatment of autoimmune diseases, can cause retinopathy as a serious side effect, which may lead to irreversible vision loss if not detected early. By employing a binary classification approach, this initiative seeks to develop a model that can predict the probability of HCQ retinopathy presence in patients. The input for this model consists of various OCT B-scan images of the retina, while the output is a quantified probability of the presence of HCQ retinopathy, providing a valuable tool for ophthalmologists to detect and mitigate this condition effectively.

## Set up environment

```bash
git clone https://github.com/shengyenlin/MEDEECS-II-2023-Spring
conda creane -n medeecsII python=3.6
pip3 install gdown
pip3 install tensorflow-gpu==1.12.3 keras==2.2.4
```

## Get and process data
```bash
bash get_data.bash  # get data drop google drive
python3 cut_img.py # cut images into top and down images
bash preprocess_data.bash # Preprocess to keras format
```

## Remark
1. After running the code above, you will see three folders:
data: The most original data, the same as on Google Drive.
data_split: Each image has been split into a top and bottom image.
data_keras: Contains three CSV files, which include the patient names and labels for train, valid, and test (column name is "HCQ_label"). There are also three subdirectories - train, valid, test - intended for use with the keras flow_from_directory function. For usage instructions, please refer to [here](https://github.com/ayushdabra/retinal-oct-images-classification/blob/master/vgg16-for-retinal-oct-images-dataset.ipynb)

## Transfer learning

1. Keras - transfer learning [tutorial](https://github.com/ayushdabra/retinal-oct-images-classification/blob/master/vgg16-for-retinal-oct-images-dataset.ipynb)
2. The model we will use is [here](https://github.com/SharifAmit/OpticNet-71)
3. What we need to do is to replace the last layer (also called the head) of the above model with a binary classification head. You can refer to train_example.py, where I have already written most of the workflow.
4. Different experiments that can be attempted with transfer learning include:
- Unfreezing the backbone (the part where machine learning extracts features).-
- Freezing the backbone.
- Varying the learning rates of the backbone and head.
