# Classification d'images par Deeplearning

> Vision appliquée pour la Robotique  
> Majeure ROBIA/ module IA Vision  
> LONCHAMBON Alexis - 5IRC
***
> Prise en main de Tensorflow

## Partie 2

### Question 1

> 1.a Expliquer la différence entre la classification d’image, la détection d’image la segmentation d’images

La classification d'images permet d'identifier une simple image et de la ranger dans une classe définie parmis une liste de classes.  

La détection d'images permet d'identifier la présence d'un ou plusieurs objets de différentes classes dans une image.

La segmentation d'image pousse un peu plus loin la détection en ajoutant aussi où se trouve l'objet identifié précisément, quels pixels correspondent a l'objet.
![comparison](https://media.licdn.com/dms/image/D4D12AQGf61lmNOm3xA/article-cover_image-shrink_720_1280/0/1656513646049?e=2147483647&v=beta&t=1WhJuMdd_Gn9GCtfxUKDGGW2IWhBlRN-46ddUHcQSNA)

> 1.b Quelles sont les grandes solutions de détection d’objets
( voir par exemple [How SSD Works](https://developers.arcgis.com/python/guide/how-ssd-works/) )

Voici une liste de solutions :

- R-CNN (Region-based Convolutional Neural Network) et plus tard Fast R-CNN, Faster R-CNN et Cascade R-CNN qui proposent des architectures plus optimisées
- SSD (Single-Shot Detector)
- YOLOv3 (You Only Look Once)
- RetinaNet

### Question 2

On s’intéresse à l’exemple suivant

[COLAB](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/hub/tutorials/tf2_object_detection.ipynb)

(Attention pour une raison curieuse si le code ne marche pas, une solution peut etre d’utiliser  une autre version de object_detection, en ajoutant par exemple une cellule avant le moment qui pose problème `!pip install object_detection==0.0.3`)

> 2.a Quelles sont les classes reconnues par le réseau ?

Les classes sont dans le fichier `mscoco_label_map.pbtxt` :
|  |  |  |  |  |
|--|--|--|--|--|
| person | bicycle | car | motorcycle | airplane |
| bus | train   | truck | boat | traffic light |
| fire hydrant | stop sign | parking meter | bench | bird |
| cat | dog | horse | sheep | cow |
| elephant | bear | zebra | giraffe | backpack |
| umbrella | handbag | tie | suitcase | frisbee |
| skis | snowboard | sports ball | kite | baseball bat |
| baseball glove | skateboard | surfboard | tennis racket | bottle |
| wine glass | cup | fork | knife | spoon |
| bowl | banana | apple | sandwich | orange |
| broccoli | carrot | hot dog | pizza | donut |
| cake | chair | couch | potted plant | bed |
| dining table | toilet | tv | laptop | mouse |
| remote | keyboard | cell phone | microwave | oven |
| toaster | sink | refrigerator | book | clock |
| vase | scissors | teddy bear | hair drier | toothbrush |

> 2.b Quelle partie du code correspond au chargement du modèle de réseau.Quelles sont les modèles proposés

La sélection du modele est dans la catégorie Model Selection. On y charge les modeles depuis le TensorFlow Hub.

On retrouve les modeles suivants :

- CenterNet HourGlass
- CenterNet ResNet
- EfficientNet
- SSD ResNet
- SSD MobileNet
- Faster R-CNN
- Mask R-CNN

> 2.c Quelles sont les structures des modèles de réseaux sous jacents ?

| Model | Network |
|--|--|
| CenterNet HourGlass   | ![Hourglass](https://www.researchgate.net/publication/354263873/figure/fig1/AS:1063041139499009@1630460166753/The-reduced-architecture-of-Hourglass-104-for-the-use-of-the-backbone-of-CenterNet.png)  </br>For the case of the CenterNet with Hourglass backbone, the stacked Hourglass Network downsamples the input by 4×, followed by two sequential hourglass modules. Each hourglass module is made up of a uniform chain of 5-layer down- and up-convolutional network with skip connections. No changes were made in this network. |
| CenterNet ResNet      | ![ResNet50](https://www.researchgate.net/publication/361971824/figure/fig1/AS:1182050384125954@1658834181499/Overall-structure-of-CenterNet-based-on-Resnet50.png)    </br>Standard ResNet modules are augmented with three transposed convolutional networks to incorporate higher resolution outputs. </br> Some modifications are by reducing the output filters of upsampling layers to 256, 128, and 64 respectively for computational reduction. The addition of a 3X3 deformable convolutional layer between each upsampling layers helped to get decent results on some standard datasets. |
| EfficientNet          | ![Efficient](https://github.com/google/automl/raw/master/efficientdet/g3doc/network.png)  SDD Avec EfficientNet |
| SSD ResNet            | ![Retinanet](https://www.researchgate.net/publication/327737749/figure/fig1/AS:672393336987655@1537322472864/The-network-architecture-of-RetinaNet-RetinaNet-uses-the-Feature-Pyramid-Network-FPN.png)  </br>RetinaNet |
| SSD MobileNet         | ![MobileNet v2 SSD](https://www.researchgate.net/publication/360288287/figure/fig3/AS:11431281090707861@1666144914309/Mobilenet-V2-SSD-network-structure.png) |
| Faster R-CNN          | ![Faster R-CNN](https://www.researchgate.net/publication/335513632/figure/fig1/AS:797773196689408@1567215360420/a-Network-structure-of-Faster-R-CNN-and-b-network-structure-of-the-proposed-FFAN-In.png) |
| Mask R-CNN            | ![MASK R-CNN](https://www.researchgate.net/publication/341717040/figure/fig3/AS:896342276706307@1590716060131/The-schematic-architecture-of-Mask-R-CNN-Cls-layer-denotes-classification-layer-Reg.jpg)  </br>Mask R-CNN with Inception Resnet v2 |

> 2.d Tester sur une douzaine d’images de votre choix (Essayer sur des images contenant le plus de classes possibles reconnus) et faites un tableau comparatif

| Original Image | CenterNet HourGlass104 | CenterNet ResNet101 | EfficientDet D4 | SSD MobileNet v1 FPN | SSD ResNet101 FPN | Faster R-CNN | Mask R-CNN |
|--|--|--|--|--|--|--|--|
| ![Z](./img/z.jpg)               | ![CNGH](./img/z1.png) | ![CNRN](./img/z2.png) | ![EDD4](./img/z3.png) | ![EEDMN](./img/z4.png) | ![SSDRN](./img/z5.png) | ![FRCNN](./img/z6.png) | ![MCNN](./img/z7.png) |
| ![Zevent](./img/zevent.jpg)     | ![CNGH](./img/zevent1.png) | ![CNRN](./img/zevent2.png) | ![EDD4](./img/zevent3.png) | ![EEDMN](./img/zevent4.png) | ![SSDRN](./img/zevent5.png) | ![FRCNN](./img/zevent6.png) | ![MCNN](./img/zevent7.png) |
| ![BL](./img/bl.jpg)             | ![CNGH](./img/bl1.png) | ![CNRN](./img/bl2.png) | ![EDD4](./img/bl3.png) | ![EEDMN](./img/bl4.png) | ![SSDRN](./img/bl5.png) | ![FRCNN](./img/bl6.png) | ![MCNN](./img/bl7.png) |
| ![Bouchon](./img/bouchon.jpg)   | ![CNGH](./img/bouchon1.png) | ![CNRN](./img/bouchon2.png) | ![EDD4](./img/bouchon3.png) | ![EEDMN](./img/bouchon4.png) | ![SSDRN](./img/bouchon5.png) | ![FRCNN](./img/bouchon6.png) | ![MCNN](./img/bouchon7.png) |
| ![Bouchon2](./img/bouchon2.jpg) | ![CNGH](./img/bouchon21.png) | ![CNRN](./img/bouchon22.png) | ![EDD4](./img/bouchon23.png) | ![EEDMN](./img/bouchon24.png) | ![SSDRN](./img/bouchon25.png) | ![FRCNN](./img/bouchon26.png) | ![MCNN](./img/bouchon27.png) |
| ![Elevage](./img/elevage.jpg)   | ![CNGH](./img/elevage1.png) | ![CNRN](./img/elevage2.png) | ![EDD4](./img/elevage3.png) | ![EEDMN](./img/elevage4.png) | ![SSDRN](./img/elevage5.png) | ![FRCNN](./img/elevage6.png) | ![MCNN](./img/elevage7.png) |
| ![Lycee](./img/l.jpg)           | ![CNGH](./img/l1.png) | ![CNRN](./img/l2.png) | ![EDD4](./img/l3.png) | ![EEDMN](./img/l4.png) | ![SSDRN](./img/l5.png) | ![FRCNN](./img/l6.png) | ![MCNN](./img/l7.png) |

### Question 3

> 3.a A quoi sert Tensorflow Hub, et y a t il des solutions équivalentes ?

Stocker les modeles en ligne

[HuggingFace](https://huggingface.co)  
[Kaggle](https://www.kaggle.com/models)

Note : TensorflowHub migre sur Kaggle a partir du 15 Novembre

> 3.b Combien trouve t’on sur tensorflow hub de réseaux de detection d’objets ?

On a 101 modeles pour l'Object Detection. Parmi ceux la :

- Inception ResNet v2
- MobileNet v2 RetinaNet
- SSD SpaghettiNet
- IRIS
- facedetection
- YOLOv5
- BlazeFace

> 3.c Quelles sont les architectures de ces réseaux ?

On a entre autres :

- EfficientNet (20)
- Faster R-CNN (15)
- SSD (15)
- CenterNet (9)
- RetinaNet (6)
- SSD MobileNet (12 en incluant V1 et V2)
- YOLO (3)

> 3.d Quelles sont les classes reconnues ?

La plupart utilisent COCO 2017 et l'autre majorité CPPE5

> 3.e Y a-t-il des exemples pour gérer une phase d’apprentissage ?

[Custom ConvNets (Kaggle)](https://www.kaggle.com/code/ryanholbrook/custom-convnets)  
[TF2 Custom Object Detection Model Tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html)
