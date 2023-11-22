# Towards Realistic Evaluation of Industrial Continual Learning Scenarios with an Emphasis on Energy Consumption and Computational Footprint

[[`Paper`](https://openaccess.thecvf.com/content/ICCV2023/html/Chavan_Towards_Realistic_Evaluation_of_Industrial_Continual_Learning_Scenarios_with_an_ICCV_2023_paper.html)] [[`Poster`](https://drive.google.com/file/d/18rZ5_DB3biaHvS2zVbjepI_h-2T9ISL3/view?usp=drive_link)] [[`Summary Video`](https://youtu.be/WvpDmG1UGSY)]

**Abstract:** Incremental Learning (IL) aims to develop Machine Learning (ML) models that can learn from continuous streams of data and mitigate catastrophic forgetting. We analyze the current state-of-the-art Class-IL implementations and demonstrate why the current body of research tends to be one-dimensional, with an excessive focus on accuracy metrics. A realistic evaluation of Continual Learning methods should also emphasize energy consumption and overall computational load for a comprehensive understanding. This paper addresses research gaps between current IL research and industrial project environments, including varying incremental tasks and the introduction of Joint Training in tandem with IL. We introduce InVar-100 (<ins>In</ins>dustrial Objects in <ins>Var</ins>ied Contexts), a novel dataset meant to simulate the visual environments in industrial setups and perform various experiments for IL. Additionally, we incorporate explainability (using class activations) to interpret the model predictions. Our approach, RECIL (<ins>R</ins>eal-world Scenarios and <ins>E</ins>nergy Efficiency considerations for <ins>C</ins>lass <ins>I</ins>ncremental <ins>L</ins>earning), provides meaningful insights about the applicability of IL approaches in practical use cases. The overarching aim is to tie the Incremental Learning and Green AI fields together and encourage the application of CIL methods in real-world scenarios. Code and dataset are available.


![Poster_img](https://github.com/Vivek9Chavan/RECIL/assets/57413096/a033df28-a033-4294-a4b0-e5641c540c42)


# InVar-100 Dataset

The **Industrial Objects in Varied Contexts** (InVar) Dataset was internally produced by our team and contains 100 objects in a total of 20,800 images (208 images per class). The objects consist of common automotive, machine, and robotics lab parts. Each class contains 4 sub-categories (52 images each) with different attributes and visual complexities.

**White background** (D<sub>wh</sub>): The object is against a clean white background, and the object is clear, centred, and in focus.

**Stationary Setup** (D<sub>st</sub>): These images are also taken against a clean background using a stationary camera setup, with uncentered objects at a constant distance. The images have lower DPI resolution with occasional cropping.

**Handheld** (D<sub>ha</sub>): These images are taken with the user holding the objects, with occasional occlusion.

**Cluttered background** (D<sub>cl</sub>): These images are taken with the object placed along with other objects from the lab in the background with no occlusion.

The dataset was produced by our staff at different workstations and labs in Berlin. Human subjects, when present in the images (e.g. holding the object), remain anonymized. More details regarding the objects used for digitization are available in the metadata file.

The InVar-100 dataset can be accessed here: http://dx.doi.org/10.24406/fordatis/266.2

<img src="https://github.com/Vivek9Chavan/RECIL/raw/main/qr-codev2.png" alt="QR Code" width="40%" />

# Efficient Data Management for Industrial Continual Learning using Self-Supervised Pretraining

Preprint will be shared soon.

**Abstract:** The availability of large quantities of data is heavily responsible for the breakthroughs and rapid adoption of Machine Learning (ML). However, data collected in miscellaneous industrial setups tend to be uncurated and heterogeneous, making it time-consuming and expensive to sort through before ML training. This becomes an incrementally compounding issue in Continual Learning projects with long timelines. We leverage the recent advancements in Computer Vision and Self-Supervised Learning (SSL) for data pruning, coreset/exemplar selection, and analysis. We present a straightforward solution to reduce the impact of catastrophic forgetting and data drift on feature representation. We study our approach on relevant industrial datasets. Our approach, RECLUSE (Representation Clustering for Continual Learning using SSL Pretrained Encoder), can be implemented before, during and after the traditional supervised ML training iteratively and in a scalable manner. We explore our implementation with varying constraints, with and without weak supervision. RECLUSE surpasses the current state-of-the-art for exemplar selection, providing an improvement in average incremental accuracy of 1.5% (with DER) to 4.1% (with PODNet) against Herding, and 0.8% on old class data against RMM. We also elaborate on Self-Supervised pertaining (using DINO and VICRegL) for Class-IL applications and observe an improvement in average incremental accuracy of 2.4% to 8.1% for different industrial datasets. Minimising the computational footprint of lifelong learning projects in industrial setups is also a focus of this work. Code implementation is available.

## RECLUSE: Representation Clustering for Continual Learning using SSL Pretrained Encoder

### Code is being added, the paper is currently under review.

## Acknowledgements

Our code borrows heavily form the following repositories:

https://github.com/G-U-N/PyCIL

https://github.com/facebookresearch/dino

https://github.com/facebookresearch/VICRegL

<a name="bibtex"></a>
## Citation

If you find our work or any of our materials useful, please cite our paper:
```
@InProceedings{Chavan_2023_ICCV,
    author    = {Chavan, Vivek and Koch, Paul and Schl\"uter, Marian and Briese, Clemens},
    title     = {Towards Realistic Evaluation of Industrial Continual Learning Scenarios with an Emphasis on Energy Consumption and Computational Footprint},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {11506-11518}
}

```
