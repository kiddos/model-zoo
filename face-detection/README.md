Face YOLO Model
===============

## Download data

WIDER face dataset: [link](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)

## Prepare data

```
python wider_prepare.py --dbname wider256x256-16x16.sqlite3 --output-size 16
```

## Train the model

```
python yolo_train.py --dbname wider256x256-16x16.sqlite --inference inference_v2 --learning-rate=1e-3 --saving True
```

## Freeze the graph

```
python yolo_freeze.py --checkpoint yolo_face-inference_v2_0/yolo-100000
```

## Run the model

```
python yolo_run.py --model yolo.pb
```

## Citation

@inproceedings{yang2016wider,
	Author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
	Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	Title = {WIDER FACE: A Face Detection Benchmark},
	Year = {2016}}
