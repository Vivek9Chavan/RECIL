# Towards Realistic Evaluation of Industrial Continual Learning Scenarios with an Emphasis on Energy Consumption and Computational Footprint
Abstract: Incremental Learning (IL) aims to develop Machine Learning (ML) models that can learn from continuous streams of data and mitigate catastrophic forgetting. We analyze the current state-of-the-art Class-IL implementations and demonstrate why the current body of research tends to be one-dimensional, with an excessive focus on accuracy metrics. A realistic evaluation of Continual Learning methods should also emphasize energy consumption and overall computational load for a comprehensive understanding. This paper addresses research gaps between current IL research and industrial project environments, including varying incremental tasks and the introduction of Joint Training in tandem with IL. We introduce InVar-100 (<ins>In</ins>dustrial Objects in <ins>Var</ins>ied Contexts), a novel dataset meant to simulate the visual environments in industrial setups and perform various experiments for IL. Additionally, we incorporate explainability (using class activations) to interpret the model predictions. Our approach, RECIL (<ins>R</ins>eal-world Scenarios and <ins>E</ins>nergy Efficiency considerations for <ins>C</ins>lass <ins>I</ins>ncremental <ins>L</ins>earning), provides meaningful insights about the applicability of IL approaches in practical use cases. The overarching aim is to tie the Incremental Learning and Green AI fields together. Code and dataset are available.
I've used the HTML <ins> tag to simulate underlining for the specific parts you indicated. 

## RECIL: Real World Scenarios and Energy Efficiency Considerations for Class Incremental Learning

# InVar-100 Dataset

<div style="display: flex; flex-direction: row;">
  <div style="flex: 0.5; padding: 10px;">
    The Industrial Objects in Varied Contexts (InVar)
Dataset was internally produced by our team and contains
100 objects in 20800 total images (208 images per class).
The objects consist of common automotive, machine and
robotics lab parts. Each class contains 4 sub-categories (52
images each) with different attributes and visual complex-
ities. White background (Dwh): The object is against a
clean white background and the object is clear, centred and
in focus. Stationary Setup (Dst): These images are also
taken against a clean background using a stationary camera
setup, with uncentered objects at a constant distance. The
images have lower DPI resolution with occasional crop-
ping. Handheld (Dha): These images are taken with the
user holding the objects, with occasional occluding. Clut-
tered background (Dcl): These images are taken with the
object placed along with other objects from the lab in the
background and no occlusion. The InVar-100 dataset can be accessed here: http://dx.doi.org/10.24406/fordatis/266
  </div>
  <div style="flex: 0.4; padding: 10px;">
    <img src="https://github.com/Vivek9Chavan/RECIL/raw/main/qr-code2.png" alt="QR Code" width="40%" />
  </div>
</div>

<a name="bibtex"></a>
## Citation

If you find our work or any of our materials useful, please cite our paper:
```
@inproceedings{chavan2023towards,
  title={Towards Realistic Evaluation of Industrial Continual Learning Scenarios with an Emphasis on Energy Consumption and Computational Footprint},
  author={ Chavan, Vivek and Koch, Paul and Schlüter, Marian and Briese, Clemens},
  booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
  year={2023}
}

```
