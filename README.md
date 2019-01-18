# Tiyuntsong Overview

Existing reinforcement learning based adaptive bitrate(ABR) approaches outperform the previous fixed control rules based methods by improving the Quality of Experience (QoE) score, while the QoE metric can hardly provide clear guidance for optimization, resulting in the unexpected strategies. In this paper, we propose Tiyuntsong, a self-play reinforcement learning approach with GAN-based method for ABR video streaming. Tiyuntsong learns strategies automatically by training two agents who are competing against each other.

Note that the competition results are evaluated with the rule rather than a numerical QoE score, and the rule has a clear optimization goal. Meanwhile, we propose GAN Enhancement Module to extract hidden features from the past status for preserving the information without the limitations of sequence lengths. Using testbed experiments, we show that the utilization of GAN significantly improves the Tiyuntsong's performance. By comparing the performance of ABRs, we observe that Tiyuntsong also betters existing ABR algorithms in the underlying metrics.

# Installation and Usage
This code is based on TensorFlow and python3. To install, run these commands:

    pip install tensorflow tflearn elo

Then, run 

    python main.py

That's all, just for fun~

# Cite
If you find this work useful in your research, please cite:

    @article{huang2018tiyuntsong,
      title={Tiyuntsong: A Self-Play Reinforcement Learning Approach for ABR Video Streaming},
      author={Huang, Tianchi and Yao, Xin and Wu, Chenglei and Zhang, Rui-Xiao and Sun, Lifeng},
      journal={arXiv preprint arXiv:1811.06166},
      year={2018}
    }
