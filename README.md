# Comparing CNN and ViT for Open-Set Face Recognition  
**Code for the paper: _Comparing CNN and ViT for Open-Set Face Recognition_**

---

## 🔧 Installation

1. **Create a general virtual environment** using Python 3.10.12 (referred to as `ENV_GENERAL`)
2. **Install the required dependencies**: `pip install -r requirements.txt`. This is the main environment used for the pre-training, fine-tuning, and evaluation phases

⚠️ To download and prepare the images from the CASIA-WebFace dataset, an additional virtual environment is required (`ENV_CASIA`) to keep dependencies isolated

---

## 📁 Repository Structure
This repository is organized into two main components:

### Pretraining (VGGFace2)

Located in the `pretraining/` directory, this component handles the pre-training phase using the VGGFace2 dataset.

Contents include:

- `main_pretraining.py` and `utils.py`: Python scripts that handle the full pre-training process
- `pretrain_all_models.sh`: Shell script to automate pre-training for all models in the paper
- `analysis_vggface2/`: Exploratory analysis of the VGGFace2 dataset

Excluded content: 

- `checkpoints/`: Intended to store model checkpoints saved during pre-training
- `logs/`: Contains training logs generated during pre-training runs
- `vggface2/`: Directory containing the VGGFace2 dataset

⚠️ The `checkpoints/` and `logs/` directories are not included in this repository due to their large size. These folders are generated automatically after running the pre-training script (`pretrain_all_models.sh`) and will contain model weights and training logs respectively.

⚠️ The VGGFace2 dataset (~83 GB) is also not included in this repository (`vggface2/`).

#### Dataset Download – VGGFace2
To download and prepare the VGGFace2 dataset for pre-training:
1. Download the dataset from Academic Torrents using this [torrent link](https://academictorrents.com/download/535113b8395832f09121bc53ac85d7bc8ef6fa5b.torrent)
2. After downloading and extracting the torrent, you will get a folder named `/VGG-Face2/data/` containing the following four files:
   - `vggface2_train.tar.gz`
   - `vggface2_test.tar.gz`
   - `train_list.txt`
   - `test_list.txt`
3. Move all four files to `/pretraining/vggface2/`
4. Navigate to that directory: `cd /pretraining/vggface2/`
5. Extract the image archives:
   - `tar -xvzf vggface2_train.tar.gz`
   - `tar -xvzf vggface2_test.tar.gz`
  
After extraction, the VGGFace2 dataset will be fully unpacked and organized within `/pretraining/vggface2/`, along with the corresponding train/test list files. The dataset is now ready for use in the pre-training stage.


### Fine-tuning and Evaluation (CASIA-WebFace)

The `CASIA-WebFace/` directory contains everything related to the fine-tuning and evaluation phases of the models using the CASIA-WebFace dataset.

Contents include:
- `main_finetuning.py`: Performs model fine-tuning for all models
- `main_evaluation.py`: Executes evaluation in both Closed-Set Recognition (CSR) and Open-Set Recognition (OSR) scenarios
- `.log` files: Log outputs from script executions
- `setup_casia_webface/`: Contains utilities and scripts necessary to download and prepare the CASIA-WebFace dataset

Excluded content:
  - `casia_webface/`:
    - `casia_webface/`: Original dataset extracted from .rec files
    - `casia_webface_imgs_mtcnn/`: Dataset cropped using MTCNN
    - `casia_webface_imgs_mtcnn_ordered/`: Dataset organized into (these last splits were created using the `split_known_unknown_dataset.py` script):
      - `train/`
      - `validation/`
      - `close_test/` (CSR scenario)
      - `open_test/` (OSR scenario)
- `finetune_evaluation_casia_webface/`: Stores fine-tuning results, metrics, and best model checkpoints

⚠️ Due to their large size, the `casia_webface/` directory and fine-tuned models in `finetune_evaluation_casia_webface/` are not included in this repository. However, the fine-tuned models can be obtained by executing the `main_finetuning.py` script.

#### Dataset Download and Preparation – CASIA-WebFace
To download and prepare the CASIA-WebFace dataset:
1. Download the dataset ZIP file from this [Google Drive link](https://drive.google.com/file/d/1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l/view)
2. Move the downloaded ZIP file to: `CASIA-WebFace/setup_casia_webface/`
3. Navigate to that directory: `cd CASIA-WebFace/setup_casia_webface/`
4. Unzip the dataset: `unzip faces_webface_112x112.zip`
5. Create a separate Python 3.10.12 virtual environment dedicated exclusively to dataset preparation (`ENV_CASIA`), activate it, and install the required packages: `pip install -r requirements.txt`
6. Convert the .rec dataset files to individual image files by running: `python3 rec_to_imgs.py` (this creates the original dataset in: `CASIA-WebFace/casia_webface/casia_webface`)
7. Deactivate the `ENV_CASIA` environment after completion
8. Activate the general environment (`ENV_GENERAL`)
9. Run MTCNN-based face cropping: `python3 mtcnn.py` (this creates: `CASIA-WebFace/casia_webface/casia_webface_imgs_mtcnn`)
10. Create train, validation, CSR and OSR evaluation splits: `python3 split_known_unknown_dataset.py` (this creates: `CASIA-WebFace/casia_webface/casia_webface_imgs_mtcnn_ordered`)

## 📝 Acknowledgement

Some parts of the dataset preparation are based on code from [Open-Set-Recognition](https://github.com/ma-xu/Open-Set-Recognition), and the CASIA-WebFace dataset is referenced from the [InsightFace dataset list](https://github.com/deepinsight/insightface/blob/master/recognition/_datasets_/README.md).

## ⚠️ Important Note

It may be necessary to modify paths in some files if you wish to replicate the work described in the article.
