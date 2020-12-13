# HackOffv3--Team--JSON-Bourne--EDCCT

Team, JSON Bourne presents EDCCT ( Embedders for Deep Contextual Clustering of Tags ), a practical approach towards accurate contextual clustering of review tags. This repository is our submission for HackOff v3.0 Hackathon for the problem statement 1 of the Siemens Healthineers challenge.

## PROJECT NAME:

EDCCT - Embedders for Deep Contextual Clustering of Tags

## TAGLINE: 

Team, JSON Bourne presents EDCCT ( Embedders for Deep Contextual Clustering of Tags ), a practical approach towards accurate contextual clustering of review tags. We use linguistic embedders ([1]: Look Up Word embeddings, [2] Contextual Word Embeddings and [3] Sentence Embeddings) to derive vectorial representation of each of the tags, following which we used cosine similarity to cluster them.

## The problem it solves:

EDCCT solves the problem of multiple tags referring to the same thing by combining similar tags into a single tag-head. We approached this problem statement while keeping in mind recent development in the field of linguistic representation of words and sentences. It is able to geometrically cluster both, contextually and meaning based tags, thereby reducing repeating review tags.

## Challenges we ran into:

While approaching this problem, we ran into two major problems, we were able to overcome both of them. The first one being, frequency based tags, did not give accurate representation of product-reviews. Secondly, lookup word embeddings did not give accurate cluster results, therefore we finally propose a deep contextual representation using Universal Sentence Encoder to embed and cluster similar tags.

## Project-Demo:

We have analysed and worked with three embeddings, namely: Universal Sentence Encoders, BERT, GloVe. We finally propose Universal Sentence Encoder as the ideal embedder.
[Please Check Out Out Demo](https://docs.google.com/document/d/1ehXetnZB1Mypdjsxc9VutyriNOssn_wdm5SDT_PEDsM/edit?usp=sharing)

```console
Product: B007FXMOV8
Orignal tags: [screen protector, rubber tip, rubber tipped, use stylus, touch screen, using stylus, stylus tip, screen use, pen great]
Generated tags after deleting similar tags: [using stylus, rubber tip]

Product: B0088U4YAG
Original tags: [car charger, works great, iphone 5, car chargers, two devices, charge phone, charger, charge, charge two devices, charging two devices, enough power charge, charger two ports]
Generated tags: [charge two devices, car charger]
```

## Technologies we used:

Python, Universal Sentence Encoder(DAN)

## Links:

GitHub: https://github.com/AmanPriyanshu/HackOffv3--Team--JSON-Bourne--EDCCT

Video Demo: 
YouTube: https://youtu.be/BzK9kgMfw88
