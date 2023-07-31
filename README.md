# MSICL

**2023.7.31. This article has been accepted by JIIS (Journal of Intelligent Information Systems).**

Improving Graph Collaborative Filtering with Multimodal-Side-Information-enriched Contrastive Learning

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

Our paper is based on the recbole framework which only provides interaction records. Multimodal information needs to be downloaded additionally.

```
python dataset/get.py
```

Because the multimodal information of some items provided by Amazon is missing, we need to filter the items that have no multimodal information. **Please overwrite the files in the original recbole library with the dataset.py file in our project**

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

