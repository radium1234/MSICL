# MSICL

Code of paper for dasfaa2023：Improving Graph Collaborative Filtering with Multimodal-Side-Information-enriched Contrastive Learning

We are constantly updating and sorting out our code

## Requirements

```
recbole==1.0.0
python==3.7.0
pytorch==1.7.1
faiss-gpu==1.7.1
cudatoolkit==10.1
```

## Datasets

Our paper is based on the recbole framework and only provides interactive information. Multimodal information needs to be downloaded separately

```
python dataset/get.py
```

Because the multimodal information of some items provided by the recbole framework is missing, we need to filter the item. **Please overwrite the files in the original recbole library with the dataset.py file in our project**

```
├─ recbole
	├─ data
		├─ dataset
        	├─ dataset.py
```

## Start training and inference

```
python main.py --model msicl
```

