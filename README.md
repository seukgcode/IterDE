#  IterDE

The codes and datasets for "IterDE: An Iterative Knowledge Distillation Framework for Knowledge Graph Embeddings".

The repo is expended on the basics of [OpenKE](https://github.com/thunlp/OpenKE).

## Folder Structure

The structure of the folder is shown below:

```csharp
IterDE
├─checkpoint
├─benchmarks
├─IterDE_FB15K237
├─IterDE_WN18RR
├─openke
├─requirements.txt
└README.md
```

Introduction to the structure of the folder:

- /checkpoint: The generated models are stored in this folder.

- /benchmarks: The datasets(FB15K237 and WN18RR) are stored in this folder.
- /IterDE_FB15K237: Training for iteratively distilling KGEs on FB15K-237.
- /IterDE_WN18RR: Training for iteratively distilling KGEs on WN18RR.
- /openke: Codes for the models of distillation for KGEs
- /requirements.txt: All the dependencies are shown in this text.
- README.md: Instruct on how to realize IterDE.

## Requirements

All experiments are implemented on CPU Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz and GPU GeForce RTX 2080 Ti.  The version of Python is 3.7.

Please run as follows to install all the dependencies:

```
pip3 install -r requirements.txt
```

## Usage

### Preparation

1. Enter openke folder.

```
cd IterDE
cd openke
```

2. Compile C++ files

```
bash make.sh
cd ../
```

### Example 1: Distill TransE on FB15K-237:

1. Firstly, we pre-train the teacher model TransE:

```
cp IterDE_FB15K237/transe_512.py ./
python transe_512.py
```

2. Then we iteratively distill the student model:

```
cp IterDE_FB15K237/transe_512_256_new.py ./
cp IterDE_FB15K237/transe_512_256_128_new.py ./
cp IterDE_FB15K237/transe_512_256_128_64_new.py ./
cp IterDE_FB15K237/transe_512_256_128_64_32_new.py ./
python transe_512_256_new.py
python transe_512_256_128_new.py
python transe_512_256_128_64_new.py
python transe_512_256_128_64_32_new.py
```

3. Finally, the distilled student will be generated in the checkpoint folder.

### Example 2: Distill ComplEx on WN18RR:

1. Firstly, we pre-train the teacher model ComplEx:

```
cp IterDE_WN18RR/com_wn_512.py ./
python com_wn_512.py
```

2. Then we iteratively distill the student model:

```
cp IterDE_WN18RR/com_512_256_new.py ./
cp IterDE_WN18RR/com_512_256_128_new.py ./
cp IterDE_WN18RR/com_512_256_128_64_new.py ./
cp IterDE_WN18RR/com_512_256_128_64_32_new.py ./
python com_512_256_new.py
python com_512_256_128_new.py
python com_512_256_128_64_new.py
python com_512_256_128_64_32_new.py
```

3. Finally, the distilled student will be generated in the checkpoint folder.

## Acknowledgement:

We refer to the code of [OpenKE](https://github.com/thunlp/OpenKE). Thanks for their contributions.
