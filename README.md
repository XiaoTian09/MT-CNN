# Comparison of single-trace and multiple-trace polarity determination for surface microseismic data using deep learning

Introduction：
We propose a multi-trace based CNN (MT-CNN) architecture to determine the P-wave first-motion polarity for 
surface microseismic data since the polarity of the recorded waveform of a microseismic event changes 
continuously across the array, while previous studies of polarity detection utilizes only individual trace
information. Field data example shows that MT-CNN significantly produces fewer
polarity prediction errors and leads to more accurate focal mechanism solutions for
microseismic events than the single-trace based CNN. In addition, the MT-CNN is also
suitable for other data, such as an earthquake dataset with a regular station distribution.
Reference：Tian et al.,2020, Comparison of single-trace and multiple-trace polarity determination for surface microseismic data using deep learning, accepted by SRL


Attached codes include the ST-CNN and MT-CNN architectures and the best models we obtained.

The traning data used in this study were accessible on the Jianguoyun via https://www.jianguoyun.com/p/DdODUCMQov3zBxir058C

Any questions, please contact: tianxiao@mail.ustc.edu.cn
