# Fusion of military knowledge Graph
by 58119125 JiangZhuoyang
## 1.Dedupe.ipynb
It is used for Property Fusion to merge more than 3000 Properties into 18. 
At the same time, it can generate entities involved in extracting properties.

The implementation process is as follows:

#### 1.1 Basic data cleaning
#### 1.2 Greedy clustering of property according to string distance
#### 1.3 Cluster special kinds of Property using trigger words
#### 1.4 Further manual screening and refinement to ensure PROPERTY quality
#### 1.5 Use the property class as a new Property and merge the values under the child property
#### 1.6 Generate entities for values under ‘countries, people and institutions’ propertiy

## 2.Fusion.ipynb
It is used for Graph Fusion to merge KG generated from different data sources.


The implementation process is as follows:

#### 2.1 Basic data cleaning
#### 2.2 Insert unstructured extracted data into structured data table (discussed by case)
#### 2.3 Fusion result analyse