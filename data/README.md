# Data

No images are stored in this repository. All three datasets are public.

## Kermany OCT2017 (training, locked test, official test)

Kermany, Goldbaum, Cai *et al.*, *Cell* 172(5):1122–1131, 2018, doi:10.1016/j.cell.2018.02.010.

* Kaggle: dataset `paultimothymooney/kermany2018` ("Retinal OCT Images"). On Kaggle add it with *Add Input*; on Colab,
  Cell 0 downloads it with `kagglehub` (needs a `KAGGLE_API_TOKEN` secret); locally:
  `kaggle datasets download -d paultimothymooney/kermany2018 && unzip kermany2018.zip`.
* Point `OCT_ROOT` at the folder that contains `train/` and `test/` with the subfolders `CNV DME DRUSEN NORMAL`
  (`.../OCT2017`). Files are named `CLASS-PATIENTID-N.jpeg`; the patient ID is used for every split.
* Expected counts: `train/` 83,484 images / 4,590 patients, `test/` 968 images / 609 patients. 546 patients appear in both;
  their 27,692 train-folder images are removed before the split.
* `python scripts/verify_split.py $OCT_ROOT` recomputes the split and checks the locked-test hash.

## Duke / Srinivasan 2014 (external validation)

Srinivasan, Kim, Mettu *et al.*, *Biomed. Opt. Express* 5(10):3568–3577, 2014, doi:10.1364/BOE.5.003568.
`2014_BOE_Srinivasan.zip` from https://people.duke.edu/~sf59/Srinivasan_BOE_2014_dataset.htm (45 Spectralis volumes:
15 normal, 15 dry AMD, 15 DME; 3,231 B-scans). Notebook 02 tries to download it; otherwise put the zip in the external
folder (`EXT_DIR`; on Colab `MyDrive/oct_qml/external`; on Kaggle add it as a dataset input). Mapping: NORMAL → NORMAL,
AMD (dry) → DRUSEN, DME → DME; one volume = one patient.

## OCTDL (external validation)

Kulyabin, Zhdanov, Nikiforova *et al.*, *Sci. Data* 11:365, 2024, doi:10.1038/s41597-024-03182-7.
Mendeley Data doi:10.17632/sncdhf53xc → *Download All* → put the zip (including `OCTDL_labels.csv`) in the external folder.
Mapping: NO → NORMAL, DME → DME, AMD with subcategory *early* or *intermediate* → DRUSEN. Late AMD (geographic atrophy and
neovascular AMD are not separated) and ERM, RAO, RVO, VID are excluded. Only images listed in `OCTDL_labels.csv` are used,
and patients come from its `patient_id` column.
