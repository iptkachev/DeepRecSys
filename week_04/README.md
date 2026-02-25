# Week 4
## Two-tower models and deep retrieval II

Lecture by [Kirill Khrylchenko](https://github.com/KhrylchenkoKirill)

Seminar by [Artem Matveev](https://github.com/matfu-pixel)

Youtube:
* [lecture](https://www.youtube.com/watch?v=Ial_HfWEZBM)
* [seminar](https://www.youtube.com/watch?v=tEcjZRDacbo)

On the fourth week of the course, we continue our discussion of two-tower models and deep retrieval. While in the previous lecture we covered the base architecture and training of two-tower models, this time we take a deep dive into the tower architecture itself:

1. Trainable (ID-based) embeddings: their advantages and limitations
2. Content encoding and inductive bias
3. Unsupervised representation learning as a powerful alternative for learning embeddings
4. User encoders
5. Additionally, we briefly discuss deep retrieval approaches that go beyond the classical two-tower setup, including multi-interest models, mixture of logits, GPU retrieval, and generative retrieval.

During the seminar, we focus on practical aspects of training neural recommender models: the main tools we use, what truly matters during training, and what the actual training code looks like in practice.