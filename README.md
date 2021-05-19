# A Robust Pedestrian and Cyclist Detection Method Using Thermal Images
A CNN implementation using the KAIST Multispectral Pedestrian Detection Benchmark Dataset

Developed with the Collaborative Robotics Lab at the University of Virginia.

We developed a pedestrian detection method via thermal imaging alone on the KAIST Multispectral Pedestrian Dataset, consisting of color and thermal pairs of pedestrians and cyclists taken from a moving vehicle. More infomation can be found in our SIEDS 2021 paper.

# Citation
If you use our extended model or functions in your research, please consider citing:

```
@INPROCEEDINGS{Anna2104:Robust,
AUTHOR="Navya Annapareddy and Emir Sahin and Sander Abraham and Md Mofijul Islam
and Max DePiro and Tariq Iqbal",
TITLE="A Robust Pedestrian and Cyclist Detection Method Using Thermal Images",
BOOKTITLE="2021 Systems and Information Engineering Design Symposium (SIEDS) (IEEE
SIEDS'21)",
ADDRESS=virtual,
DAYS=30,
MONTH=apr,
YEAR=2021,
KEYWORDS="Pedestrian Detection; Thermal Image; Deep Learning",
ABSTRACT="Computer vision techniques have been frequently applied to pedestrian and
cyclist detection for the purpose of providing sensing capabilities to
autonomous vehicles, and delivery robots among other use cases. Most
current computer vision approaches for pedestrian and cyclist detection
utilize RGB data alone. However, RGB-only systems struggle in poor lighting
and weather conditions, such as at night, or during fog or precipitation,
often present in pedestrian detection contexts. Thermal imaging presents a
solution to these challenges as its quality is independent of time of day
and lighting conditions. The use of thermal imaging input, such as those in
the Long Wave Infrared (LWIR) range, is thus beneficial in computer vision
models as it allows the detection of pedestrians and cyclists in variable
illumination conditions that would pose challenges for RGB-only detection
systems. In this paper, we present a pedestrian and cyclist detection
method via thermal imaging using a deep neural network architecture. We
have evaluated our proposed method by applying it to the KAIST Pedestrian
Benchmark dataset, a multispectral dataset with paired RGB and thermal
images of pedestrians and cyclists. The results suggest that our method
achieved an F1-score of 81.34\%, indicating that our proposed approach can
successfully detect pedestrians and cyclists from thermal images alone."
}
```
