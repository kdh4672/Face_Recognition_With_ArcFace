# Face_Recognition_With_ArcFace
- ## Updated : 2020-08-15 
- TODO : ArcFace Loss implementation
- Code Reference: [[Repo1](https://github.com/TreB1eN/InsightFace_Pytorch)], [[Repo2](https://github.com/ronghuaiyang/arcface-pytorch)] ,*[[Repo3](https://github.com/wujiyang/Face_Pytorch)]
- ArcFace Paper Review : [[ArcFace](https://github.com/kdh4672/DH_Lab/blob/master/Paper_Review/Arcface.pdf)]
- Presentation File : [[Arcface.pptx](https://github.com/kdh4672/dlstudy/files/5078083/Arcface.pptx)]
- Test1 : Center is Kong,  Not perfect but almost
![result](https://user-images.githubusercontent.com/54311546/90228642-89923880-de51-11ea-9c1c-af1dea9c4466.gif)



- ## Updated : 2020-08-22
- TODO : Upgrade Face recognition performance
- What's New : Extract Featrues'degree Compared to Center, and the second nearest degree, Apply Thresh Hold at degree 30
- Model is Trained with Three class people
- Kong Result: 

![Kong_Output](https://user-images.githubusercontent.com/54311546/90952351-e61ed480-e49d-11ea-8619-442ba044466b.gif)
- June Result:

![June_Output](https://user-images.githubusercontent.com/54311546/90952358-f6cf4a80-e49d-11ea-8ec7-8deacfdae78a.gif)
- Sim Result:

![output_sim](https://user-images.githubusercontent.com/54311546/92383637-aa486800-f149-11ea-9c24-0bb817e408c4.gif)


- Three People:

![output](https://user-images.githubusercontent.com/54311546/92383726-d19f3500-f149-11ea-886c-e5c305c9fad6.gif)


- ## Updated : 2020-09-09
- TODO: Test Open-Set Recognition
- If angle between Centre and Feature of Input image is larger than 40, Output is 'Unknown'

- Kim is not enrolled in gallery (403 IDs are enrolled in gallery), that means Kim is New ID who was not trained.

![HS_Output](https://user-images.githubusercontent.com/54311546/92565133-2ca16b00-f2b5-11ea-898f-0276e3d6d1f9.gif)

- ## Updated : 2020-11-30
- TODO: Face Data Augmentation
- Used: [[DepthNet](https://github.com/kdh4672/DepthNets)]
- Result (Train Face Recognition With Augmented Data) : Not good, should be more developed
- Link: [[Augmented Face](https://github.com/kdh4672/face_augment)]
## Input

<img src="https://user-images.githubusercontent.com/54311546/102696484-0dca5180-4272-11eb-9f2d-f711c6b28386.jpg" width="300" height="300">

## Output (generated 30 images to gif)
![wb](https://user-images.githubusercontent.com/54311546/102696447-b4622280-4271-11eb-9472-1e13eee22b20.gif)
