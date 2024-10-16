# TGAF
The Mindspore code of "Compressed Video Quality Enhancement with Temporal Group Alignment and Fusion".

# Requirements
CUDA==11.6 mindspore==2.2.14 python==3.7.5

# Dataset 
Training Data: [MFQEv2](https://github.com/ryanxingql/mfqev2.0)

Test Data: [JCT-VC](https://ieeexplore.ieee.org/document/6317156) and [MFQEv2](https://github.com/ryanxingql/mfqev2.0)

# Train and Test
```python
python train_SPL.py
python test_SPL.py
```


# Acknowledgments
Thanks for the support of Huawei MindSpore Platform for training our model. 

# Citation
If this repository is helpful to your research, please cite our paper:
```python
@inproceedings{zhu2024cpga,
  title={CPGA: Coding Priors-Guided Aggregation Network for Compressed Video Quality Enhancement},
  author={Zhu, Qiang and Hao, Jinhua and Ding, Yukang and Liu, Yu and Mo, Qiao and Sun, Ming and Zhou, Chao and Zhu, Shuyuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2964--2974},
  year={2024}
}
@article{zhu2024deep,
  title={Deep Compressed Video Super-Resolution With Guidance of Coding Priors},
  author={Qiang Zhu, Feiyu Chen, Yu Liu, Shuyuan Zhu, Bing Zeng},
  journal={ IEEE Transactions on Broadcasting },
  volume={70},
  issue={2},
  pages={505-515},
  year={2024}
  publisher={IEEE},
  doi={10.1109/TBC.2024.3394291}
}
@article{zhu2024compressed,
  title={Compressed Video Quality Enhancement with Temporal Group Alignment and Fusion},
  author={Qiang, Zhu and Yajun, Qiu and Yu, Liu and Shuyuan, Zhu and Bing, Zeng},
  journal={IEEE Signal Processing Letters},
  year={2024},
  publisher={IEEE},
  doi={10.1109/LSP.2024.3407536}
}
```
