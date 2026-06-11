# Assessing Peruvian Poverty from Satellite Imagery

This repository contains the official implementation of the research project: **Assessing Peruvian Poverty from Satellite Imagery**.

This project investigates the use of Vision Transformers (ViT) for predicting the wealth index of Peruvian households by integrating socioeconomic data from the Encuesta Nacional de Hogares del Perú (ENAHO) with Sentinel-2 satellite images, constructing a novel dataset (ENAHO-S2) covering over 5,000 geographically referenced household clusters.

## Requirements

### Data

#### Encuesta Nacional de Hogares del Perú (ENAHO)

Follow these steps to download the housing and household characteristics of a specific year:

1. Go to ["Sistema de Microdatos"][inei-microdatos] of the Peruvian Institute of Statistics and Informatics (INEI).

2. Navigate to the "Consulta por Encuestas" tab.

3. From the "ENCUESTA" dropdown, select "ENAHO Metodología ACTUALIZADA".

4. From the next dropdown, select "Condiciones de Vida y Pobreza - ENAHO".

5. Choose your target year from the "AÑO" dropdown.

6. From the "Período" dropdown, select "Anual - (Ene-Dic)".

7. Look for the first row in the results table and click "SPSS" under the "Descarga" column.

8. Save the downloaded ZIP file in the `data/raw` directory of this repository.

### Dependencies

> *This project uses a [`conda`][conda] environment to manage the dependencies.*

Run the following command to create an environment with the required dependencies:

```sh
conda env create --file environment.yml
```

## Training

To train the ViT-B/16 model from scratch using the ENAHO-S2 dataset, run the following command. The default configuration uses an Adam optimizer, a batch size of 32, and trains for 50 epochs:

```py
python train.py --batch-size 32 --epochs 50 --lr 1e-4
```

**Hardware note:** The original training was performed on an NVIDIA L40S GPU using the Lightning AI platform.

## Evaluation

To evaluate the trained model on the validation set and compute the coefficient of determination, run:

```py
python eval.py --model-file checkpoints/best_model.pth --batch-size 32
```

[inei-microdatos]: https://proyectos.inei.gob.pe/microdatos/
[conda]: https://docs.conda.io
