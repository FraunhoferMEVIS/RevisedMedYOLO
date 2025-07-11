# RevisedMedYOLO

RevisedMedYOLO is a 3D bounding box detection model for medical data, which fixed some issues found in the original MedYOLO.

### Set up environment:

```bash
$ conda create --name MedYOLO python=3.12.9
$ conda activate MedYOLO
$ pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
$ pip install -r requirements.txt
```

### Build Docker image

```bash
$ docker build -t medyolo:latest .
```

### Preparing your data for use with MedYOLO:

Here is a description of what format MedYOLO labels need to have and how the folders containing the data need to be organized. Pleaes not that the label coordinate order is `ZXY` and they use relative image coordinates.

Label format: `Class-number Z-Center X-Center Y-Center Z-Length X-Length Y-Length`

Center positions and edge lengths should be given as fractions of the image extents.
Center coordinates should fall between -0.5 and 1.5.

Example label entry: `1 0.142 0.308 0.567 0.239 0.436 0.215`

Image and label files should have the same filename except for their extension (e.g. image `CT_001.nii.gz` corresponds to label `CT_001.txt`).

Data yaml files can be modeled after `/data/example.yaml` and should be saved in the `/data/` folder.
Your NIfTI images and text labels should be organized into folders as follows:

```
|--- parent_directory
|   |--- images
|   |   |--- train
|   |   |   |--- train_img001.nii.gz
|   |   |   |--- ...
|   |   |--- val
|   |   |   |--- val_img001.nii.gz
|   |   |   |--- ...
|   |--- labels
|   |   |--- train
|   |   |   |--- train_img001.txt
|   |   |   |--- ...
|   |   |--- val
|   |   |   |--- val_img001.txt
|   |   |   |--- ...
```

Here `/parent_directory/` is the path to your data.
It's recommended to store your data outside the MedYOLO directory.

### Example Training call:

```bash
$ python train.py --data example.yaml --adam --norm CT --epochs 1000 --patience 200 --device 0
```

This trains a MedYOLO model on the data found in `example.yaml` using Adam as the optimizer and the default CT normalization.
The model will train on `GPU:0` for up to 1000 epochs or until 200 epochs have passed without improvement.

By default, the small version of the model will be trained.
To train a larger model append `--cfg /MedYOLO_directory/models3D/yolo3Dm.yaml` or the yaml file that corresponds to the model you'd like to train.
Larger models can be generated by modifying the `depth_multiple` and `width_multiple` parameters in the model yaml file.

By default, MedYOLO will use the hyperparameters for training from scratch.
To train using different hyperparameters append `--hyp /MedYOLO_directory/data/hyps/hyp.finetune.yaml` or the yaml file that corresponds to the hyperparameters you'd like to use.

### Example Inference call:

```bash
$ python detect.py --source /path_to_images/ --weights /path_to_model_weights/model_weights.pt --device 0 --save-txt
```

This runs inference on the images in `/path_to_images/` with the model saved in `model_weights.pt` using `GPU:0`.
The model weights specify the model size so a model configuration yaml is not required.
Model predictions will be saved as txt files in the `/runs/detect/exp/` directory.

By default, labels are not saved during inference.
Using the `--save-txt` argument will save labels in the default directory, which can be changed by specifying the `project` and `name` arguments

By default, confidence levels are printed to the screen but not saved in .txt labels.
Use the `--save-conf` argument to append the model's confidence level at the end of each saved label entry.
Configuring the `max_det`, `conf-thresh`, and `iou-thresh` arguments alongside `--save-conf` can be helpful when troubleshooting trained models.

### Converting MedYOLO predictions into NIfTI masks:

`/utils3D/nifti_utils.py` contains an example script for converting MedYOLO predictions into viewable NIfTI masks.
This can also be useful for verifying that your MedYOLO labels mark the correct positions before you begin training a model.
As with the label creation process, there are several ways you may want to use MedYOLO's predicted bounding boxes, so this can also be used as a schematic for interpreting your model's output.

### Loss plots

To plot the metrics logged during a training run, you can use the script `plot_loss_curves.py`.

```bash
$ python plot_loss_curves.py <path_to_model_output_folder>
```

### ONNX export

It is possible to export trained models to the ONNX format with the script `onnx_export.py`

```bash
$ python onnx_export.py --weights <path_to_weights> --output <path_for_onnx_model> --img-size-x 256 --img-size-y 256 --img-size-z 256 --input-channels 2
```


### Additional Details:

MedYOLO and RevisedMedYOLO use simple rescaling to a fixed image size during training and inference. To enable fitting the image dimensions properly to the training data, we added separate options `--img-size-x`, `--img-size-y` and `--img-size-z` for the image extents. Multiples of 64 should work fine for these image extents.

Training on 3D images is usually very memory intensive. As RevisedMedYOLO (as well as MedYOLO and YOLOV5) uses gradient accumulation with a nominal batch size of 64, you can safely reduce the batch sizes to as low 2 or even 1 and should still get good results.

RevisedMedYOLO works for 3D images with one or mutliple channels. For that you have to configure the parameter `ch` in the model configuration file. 

### Attribution

If you use this work, please cite our corresponding publication (https://openreview.net/pdf?id=Fl44mi5dFn):

```bibtex
@inproceedings{geissler2025revisedmedyolo,
    title={RevisedMedYOLO: Unlocking Model Performance by Careful Training Code Inspection},
    author={Geissler, Kai and Moltz, Jan Hendrik and Meine, Hans and Wenzel, Markus},
    booktitle={Medical Imaging with Deep Learning-Short Papers},
    year={2025}
}
```

Please also cite the original MedYOLO paper, as we strongly built on their work. See their publication in JIIM: https://doi.org/10.1007/s10278-024-01138-2

