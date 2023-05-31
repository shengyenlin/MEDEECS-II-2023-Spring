# Special Topic in innovative integration of medicine and EECS (II)

The course aims to train students to solve pratical medical problems with artificial intelligence techniques. The course is still ongoing, and a project will be finished at the end of the course.

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
1. 在跑完上面的code之後，你會看到三個資料夾
- `data`: 最原始的data，和google drive上面一樣
- `data_split`: 每張圖片都被切分成top和down的image
- `data_keras`: 裡面會有三個csv, 會含有是train, valid和test的patient name和label (column name 為 "HCQ_label")。裡面還會有三個subdirectories - train, valid, test - 是拿來給keras `flow_from_directory`這個function用的，使用說明請參考[這裡](https://github.com/ayushdabra/retinal-oct-images-classification/blob/master/vgg16-for-retinal-oct-images-dataset.ipynb)

## Transfer learning

1. keras - transfer learning的[tutorial](https://github.com/ayushdabra/retinal-oct-images-classification/blob/master/vgg16-for-retinal-oct-images-dataset.ipynb)
2. 我們要用的model在[這裡](https://github.com/ayushdabra/retinal-oct-images-classification)
3. 我們要做的就是把上面這個model的最後一層（又叫做head），換成一個binary classificatio的head，可以參考`train_example.py`，我已經把大部分的workflow都寫好了
4. Transfer learning可以嘗試的不同實驗
    - 不鎖backbone (machine learning extract feature的部分)
    - 鎖backbone
    - backbone和head的learning rate大小不一樣

